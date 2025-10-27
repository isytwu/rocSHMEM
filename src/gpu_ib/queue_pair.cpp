/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "queue_pair.hpp"

#include <hip/hip_runtime.h>

#include "gda_device.hpp"
#include "endian.hpp"
#include "gpuib_macros.inl"
#if !defined(GPUIB_IONIC) && !defined(GPUIB_BNXT)
#include "segment_builder.hpp"
#endif
#include "util.hpp"

namespace rocshmem {

QueuePair::QueuePair(struct ibv_pd* pd) {
  allocator.allocate((void**)&nonfetching_atomic, 8);
  CHECK_HIP(hipMemset(nonfetching_atomic, 0, 8));
  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

  ibv_mr *mr = ibv_reg_mr(pd, nonfetching_atomic, 8, access);
  GPUIB_CHECK_NNULL(mr, "ibv_reg_mr");

#if defined(GPUIB_IONIC) || defined(GPUIB_BNXT)
  nonfetching_atomic_lkey = mr->lkey;
#else
  nonfetching_atomic_lkey = htobe32(mr->lkey);
#endif

  allocator.allocate((void**)&fetching_atomic, 8 * FETCHING_ATOMIC_CNT);
  CHECK_HIP(hipMemset(fetching_atomic, 0, 8 * FETCHING_ATOMIC_CNT));
  access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  mr = ibv_reg_mr(pd, fetching_atomic, 8 * FETCHING_ATOMIC_CNT, access);
  GPUIB_CHECK_NNULL(mr, "ibv_reg_mr");
#if defined(GPUIB_IONIC) || defined(GPUIB_BNXT)
  fetching_atomic_lkey = mr->lkey;
#else
  fetching_atomic_lkey = htobe32(mr->lkey);
#endif

  allocator.allocate((void**)&fetching_atomic_freelist, sizeof(FreeListT*));
  new (fetching_atomic_freelist) FreeListT();
  for(int i{0}; i < FETCHING_ATOMIC_CNT; i+=__AMDGCN_WAVEFRONT_SIZE) {
    fetching_atomic_freelist->push_back(fetching_atomic + i);
  }
}


/******************************************************************************
 ************************ PROVIDER-SPECIFIC HELPERS ***************************
 *****************************************************************************/
#ifdef GPUIB_IONIC
__device__ uint64_t QueuePair::get_same_qp_lane_mask() {
  uint64_t lane_mask = get_active_lane_mask();
  uintptr_t this_val = reinterpret_cast<uintptr_t>(this);

  // exclude threads operating on a different qp from this thread lane mask
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    uint64_t bit_i = 1ull << i;
    if ((lane_mask & bit_i) && __shfl(this_val, i) != this_val) {
      lane_mask &= ~bit_i;
    }
  }

  return lane_mask;
}

__device__ bool QueuePair::cq_lock_try_acquire(uint64_t activemask) {
  uint32_t cq_lock_val = SPIN_LOCK_INVALID;

  if (is_first_active_lane(activemask)) {
    cq_lock_val = SPIN_LOCK_UNLOCKED;
    __hip_atomic_compare_exchange_strong(&cq_lock, &cq_lock_val, SPIN_LOCK_LOCKED,
                                         __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  }
  cq_lock_val = __shfl(cq_lock_val, get_first_active_lane_id(activemask));

  return (cq_lock_val == SPIN_LOCK_UNLOCKED);
}

__device__ void QueuePair::cq_lock_release(uint64_t activemask) {
  if (is_first_active_lane(activemask)) {
    __hip_atomic_store(&cq_lock, SPIN_LOCK_UNLOCKED, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
}

__device__ uint32_t QueuePair::reserve_sq(uint64_t activemask, uint32_t num_wqes) {
  uint32_t my_sq_prod = 0;

  // reserve space for wqes in sq
  if (is_first_active_lane(activemask)) {
    my_sq_prod = __hip_atomic_fetch_add(&sq_prod, num_wqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  my_sq_prod = __shfl(my_sq_prod, get_first_active_lane_id(activemask));

  // wait for that space to be available
  quiet_internal(activemask, my_sq_prod + num_wqes - sq_mask);

  return my_sq_prod;
}

__device__ uint32_t QueuePair::commit_sq(bool last, uint32_t my_sq_prod, uint32_t num_wqes, struct ionic_v1_wqe *wqe) {
  uint32_t dbprod = my_sq_prod + num_wqes;

  if (last) {
    // signal last wqe before the doorbell
    wqe->base.flags |= swap_endian_val<uint16_t>(IONIC_V1_FLAG_SIG);

    while (__hip_atomic_load(&sq_dbprod, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != my_sq_prod) {
      // spin
    }

    ring_doorbell(dbprod);

    __hip_atomic_exchange(&sq_dbprod, dbprod, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }

  return dbprod;
}

__device__ void QueuePair::poll_wave_cqes(uint64_t activemask) {
  uint32_t my_logical_lane_id = get_active_lane_num(activemask);
  uint32_t my_cq_pos = cq_pos + my_logical_lane_id;

  /* Look at the cqe at the current position in the cq buffer */
  struct ionic_v1_cqe *cqe = &cq_buf[my_cq_pos & cq_mask];

  /* Determine expected color based on cq wrap count */
  uint32_t qtf_color_bit = swap_endian_val<uint32_t>(IONIC_V1_CQE_COLOR);
  uint32_t qtf_color_exp = qtf_color_bit;
  if (my_cq_pos & (cq_mask + 1)) {
    qtf_color_exp = 0;
  }

  /* Wait for at least one thread cqe color == expected color */
  uint32_t qtf_be;
  bool ready;
  uint64_t ballot_ready;
  do {
    qtf_be = *(volatile uint32_t *)(&cqe->qid_type_flags);
    ready = (qtf_be & qtf_color_bit) == qtf_color_exp;
    ballot_ready = __ballot(ready);
  } while (!ballot_ready);

  /* Other threads saw a ready cqe, but not this thread */
  if (!ready) {
    return;
  }

  uint32_t msn = swap_endian_val<uint32_t>(cqe->send.msg_msn);

  /* Report if the completion indicates an error. */
  if (!!(qtf_be & swap_endian_val<uint32_t>(IONIC_V1_CQE_ERROR))) {
#ifdef DEBUG
    uint32_t qtf = swap_endian_val<uint32_t>(qtf_be);
    uint32_t qid = qtf >> IONIC_V1_CQE_QID_SHIFT;
    uint32_t type = (qtf >> IONIC_V1_CQE_TYPE_SHIFT) & IONIC_V1_CQE_TYPE_MASK;
    uint32_t flag = qtf & 0xf;
    uint32_t status = swap_endian_val<uint32_t>(cqe->status_length);
    uint64_t npg = swap_endian_val<uint64_t>(cqe->send.npg_wqe_id);

    printf("QUIET ERROR: qid %u type %u flag %#x status %u msn %u npg %lu\n",
        qid, type, flag, status, msn, npg);
#endif
    /* No other way to signal an error, so just crash. */
    abort();
  }

  /* Only proceed with the furthest ahead cqe to update the sq state */
  uint64_t my_lane_mask = 1ull << __lane_id();
  uint64_t lesser_lane_mask = my_lane_mask - 1;
  if (my_lane_mask != (ballot_ready & ~lesser_lane_mask)) {
    return;
  }

  /* update position in the cq */
  cq_pos = my_cq_pos + 1;

  /*
   * Ring cq doorbell frequently enough to avoid cq full.
   *
   * NB: IONIC_CQ_GRACE is 100
   */
  if (((cq_pos - cq_dbpos) & cq_mask) >= 100) {
    cq_dbpos = cq_pos;
    __atomic_store_n(cq_dbreg, cq_dbval | (cq_mask & cq_dbpos), __ATOMIC_SEQ_CST); //TODO:maybe relaxed?
  }

  sq_msn = msn;
}

__device__ void QueuePair::quiet_internal(uint64_t activemask, uint32_t cons) {
  /* wait for sq_msn to catch up or pass cons. */
  /* 0x800000 - sign bit for 24-bit fields     */
  while ((sq_msn - cons) & 0x800000) {
    if (!cq_lock_try_acquire(activemask)) {
      continue;
    }

    /* with lock acquired, this wave polls cqes until caught up */
    while ((sq_msn - cons) & 0x800000) {
      poll_wave_cqes(activemask);
    }

    cq_lock_release(activemask);
    break;
  }
}
#endif // GPUIB_IONIC

#ifndef GPUIB_BNXT
#ifdef GPUIB_IONIC
__device__ void QueuePair::ring_doorbell(uint32_t pos) {
  // TODO When threads write at once to the same address, not all writes reach the bus.
  for (int i = 0; i < 64; ++i) {
    if (__lane_id() == i) {
      __threadfence();
      __atomic_store_n(sq_dbreg, sq_dbval | (sq_mask & pos), __ATOMIC_SEQ_CST);
    }
  }
  __threadfence();
}
#else // !GPUIB_IONIC
__device__ void QueuePair::ring_doorbell(uint64_t db_val, uint64_t my_sq_counter) {
  swap_endian_store(const_cast<uint32_t*>(dbrec), (uint32_t)my_sq_counter);//告诉NIC，队列中从旧的队列尾部到这个新的索引之间的所有WQE都是新提交的、需要处理的。另外，这是持续递增的值，没有取模
  __atomic_signal_fence(__ATOMIC_SEQ_CST);//插入一个编译器层面的强序 fence，保证前面的内存写入（doorbell record）不会被编译器重排到后面。

  __hip_atomic_store(db.ptr, db_val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);//db_val是最后一个WQE的字段
  uint64_t db_uint = __hip_atomic_load(&db.uint, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); // 以 relaxed 顺序读取 doorbell 寄存器的当前值。
  db_uint ^= 0x100;//0x100=0b1 0000 0000，这里把第9个bit 翻转了，TODO 为什么？uint和ptr是union
  __hip_atomic_store(&db.uint, db_uint, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); // db_val是wqe的数值，不会影响到sq_buf里WQE的值，只是改doorbell寄存器
}
#endif // !GPUIB_IONIC
#endif // !GPUIB_BNXT

#ifndef GPUIB_BNXT
#ifdef GPUIB_IONIC
__device__ void QueuePair::quiet() {
  quiet_internal(get_same_qp_lane_mask(), sq_prod);
}
#else // !GPUIB_IONIC
__device__ void QueuePair::quiet() {
  constexpr size_t BROADCAST_SIZE = 1024 / __AMDGCN_WAVEFRONT_SIZE; // 假设最大1024 thr（16 warp）？BROADCAST_SIZE相当于是最大的warp数
  __shared__ uint64_t wqe_broadcast[BROADCAST_SIZE];
  uint8_t wavefront_id = get_flat_block_id() / __AMDGCN_WAVEFRONT_SIZE;//get_flat_block_id返回的是thr id；wavefront_id是block内的warp id
  wqe_broadcast[wavefront_id] = 0;

  uint64_t activemask = get_active_lane_mask();
  uint8_t num_active_lanes = get_active_lane_count(activemask);
  uint8_t my_logical_lane_id = get_active_lane_num(activemask);
  bool is_leader{my_logical_lane_id == 0};
  const uint64_t leader_phys_lane_id = get_first_active_lane_id(activemask);

  while (true) {
    bool done{false};
    uint64_t quiet_amount{0};
    uint64_t wave_cq_consumer{0};
    while (!done) {
      uint64_t active = __hip_atomic_load(&quiet_active, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint64_t posted = __hip_atomic_load(&quiet_posted, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); // quiet_posted表示成功ring DB的wqe个数
      uint64_t completed = __hip_atomic_load(&quiet_completed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (!(posted - completed)) {
        return;//这里说明别的线程进来可能poll了cqe
      }
      int64_t quiet_val = posted - active;
      if (quiet_val <= 0) {
        continue;
      }
      quiet_amount = min(num_active_lanes, quiet_val);//quiet实际个数
      if (is_leader) {
        done = __hip_atomic_compare_exchange_strong(&quiet_active, &active, active + quiet_amount, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); // 返回的bool，而不是old value；另外还有个区别：如果不相等，则将 *expected 更新为 *addr 的当前值，返回 false；；；多个warp的leader在CAS抢quiet_active的推进权
        if (done) {
          wave_cq_consumer = __hip_atomic_fetch_add(&cq_consumer, quiet_amount, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); // 注意有fetch是返回旧值，注意这里的数值依然有可能和quiet_active不同，线程快慢的问题，可能thrA在293和295之间别的线程B先进来更新，这里才是实际的warp在CQE上取号
        }
      }
      done = __shfl(done, leader_phys_lane_id);
    }
    wave_cq_consumer = __shfl(wave_cq_consumer, leader_phys_lane_id);
    uint64_t my_cq_consumer = wave_cq_consumer + my_logical_lane_id;
    uint64_t my_cq_index = my_cq_consumer % cq_cnt;

    if (my_logical_lane_id < quiet_amount) {
      volatile mlx5_cqe64 *cqe_entry = &cq_buf[my_cq_index];
      uint16_t be_wqe_counter{0};
      uint8_t op_own{0};
      uint8_t owner_bit = (my_cq_consumer >> cq_log_cnt) & 1; // 应该等同my_cq_consumer/cq_cnt
      bool vote_failed{true};

      while (vote_failed) {//整个warp去poll cqe
        op_own = *((volatile uint8_t*)&cqe_entry->op_own);
        bool my_ownership_vote = (op_own & 1) == owner_bit;
        bool my_opcode_vote = (op_own >> 4) != MLX5_CQE_INVALID;
        uint64_t votes = __ballot(my_ownership_vote && my_opcode_vote);
        vote_failed = __popcll(votes) < quiet_amount;//等待warp的所有thr都poll到
        if (!vote_failed) {
          be_wqe_counter = *((volatile uint16_t*)&cqe_entry->wqe_counter);
        }
      }

      uint16_t wqe_counter;
      swap_endian_store(const_cast<uint16_t*>(&wqe_counter), reinterpret_cast<uint16_t>(be_wqe_counter));
      uint64_t wqe_id = outstanding_wqes[wqe_counter]; // post wqe的时候存入了 my_sq_counter。这里是不是说明硬件返回的是%OUTSTANDING_TABLE_SIZE的值
      __hip_atomic_fetch_max(&wqe_broadcast[wavefront_id], wqe_id, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP); // 拿到warp里最大的wqe_id
      uint8_t mlx5_invld_bits = MLX5_CQE_INVALID << 4 | owner_bit; // owner_bit不变
      *((volatile uint8_t*)&cqe_entry->op_own) = mlx5_invld_bits;
      __atomic_signal_fence(__ATOMIC_SEQ_CST);
    }
    if (is_leader) {
      uint64_t completed {0};
      do {
        completed = __hip_atomic_load(&quiet_completed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      } while (completed != wave_cq_consumer);//这里也是等待前面的先ringDB，和437行类似，严格按照CQE idx顺序去ringDB

      swap_endian_store(const_cast<uint32_t*>(cq_dbrec), (uint32_t)(wave_cq_consumer + quiet_amount));//UpdateCqDbrRecord?
      __atomic_signal_fence(__ATOMIC_SEQ_CST);

      uint64_t sunk_wqe_id = wqe_broadcast[wavefront_id]; // warp里最大的wqe_id
      __hip_atomic_fetch_max(&sq_sunk, sunk_wqe_id, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_fetch_add(&quiet_completed, quiet_amount, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); // sq_sunk是wqe idx，比如poll到第一个CQE之后，sq_sunk=0， quiet_completed=1
    }
  }
}
#endif // !GPUIB_IONIC
#endif // !GPUIB_BNXT

#ifndef GPUIB_BNXT
#ifdef GPUIB_IONIC//IONIC 是 Pensando 公司推出的智能网络适配器（SmartNIC）品牌，支持高性能 RDMA、加速网络和存储等功能。
__device__ void QueuePair::post_wqe_rma(int pe, int32_t size, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode) {
  uint64_t activemask = get_same_qp_lane_mask();
  uint32_t num_wqes = get_active_lane_count(activemask);
  uint32_t my_logical_lane_id = get_active_lane_num(activemask);
  uint32_t my_sq_prod = reserve_sq(activemask, num_wqes);
  uint32_t my_sq_pos = my_sq_prod + my_logical_lane_id;
  struct ionic_v1_wqe *wqe = &sq_buf[my_sq_pos & sq_mask];

  // TODO why is this needed?
  if (size && !laddr && opcode == IONIC_V2_OP_RDMA_WRITE) {
    size = 1;
  }

  wqe->base.wqe_id = my_sq_pos;
  wqe->base.op = opcode;
  wqe->base.num_sge_key = size ? 1 : 0;
  wqe->base.flags = swap_endian_val<uint16_t>(0);
  wqe->base.imm_data_key = swap_endian_val<uint32_t>(0);

  wqe->common.rdma.remote_va_high = swap_endian_val<uint32_t>(reinterpret_cast<uint64_t>(raddr) >> 32);
  wqe->common.rdma.remote_va_low = swap_endian_val<uint32_t>(reinterpret_cast<uint64_t>(raddr));
  wqe->common.rdma.remote_rkey = swap_endian_val<uint32_t>(rkey);
  wqe->common.length = swap_endian_val<uint32_t>(size);

  if (size) {
    if (opcode == IONIC_V2_OP_RDMA_WRITE && size <= inline_threshold) {
      wqe->base.flags |= swap_endian_val<uint16_t>(IONIC_V1_FLAG_INL);
      wqe->base.num_sge_key = 0;
      if (!laddr) {
        // TODO why is this needed?
        wqe->common.pld.data[0] = 1;
      } else {
        memcpy(wqe->common.pld.data, laddr, size);
      }
    } else {
      wqe->common.pld.sgl[0].va = swap_endian_val<uint64_t>(reinterpret_cast<uint64_t>(laddr));
      wqe->common.pld.sgl[0].len = swap_endian_val<uint32_t>(size);
      wqe->common.pld.sgl[0].lkey = swap_endian_val<uint32_t>(lkey);
    }
  }

  commit_sq(is_last_active_lane(activemask), my_sq_prod, num_wqes, wqe);
}
#else // !GPUIB_IONIC
__device__ void QueuePair::post_wqe_rma(int pe, int32_t size, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode) {
  uint64_t activemask = get_active_lane_mask();
  uint8_t num_active_lanes = get_active_lane_count(activemask);
  uint8_t my_logical_lane_id = get_active_lane_num(activemask);
  bool is_leader{my_logical_lane_id == 0};
  const uint64_t leader_phys_lane_id = get_first_active_lane_id(activemask);
  uint8_t num_wqes{num_active_lanes};//每个thr处理一个wqe
  uint64_t wave_sq_counter{0};

  if (is_leader) {
    wave_sq_counter = __hip_atomic_fetch_add(&sq_posted, num_wqes, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);//返回的是old value
  }
  wave_sq_counter = __shfl(wave_sq_counter, leader_phys_lane_id);
  uint64_t my_sq_counter = wave_sq_counter + my_logical_lane_id;
  uint64_t my_sq_index = my_sq_counter % sq_wqe_cnt;

  while (true) {
    uint64_t db_touched = __hip_atomic_load(&sq_db_touched, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t sunk = __hip_atomic_load(&sq_sunk, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    int64_t num_active_sq_entries = db_touched - sunk; // 当前队列中还未完成的 WQE 数量；sunk是index！相当于db_touched-quiet_completed+1，多了1，相对应的num_free_entries少了1
    if (num_active_sq_entries < 0) {//TODO 啥时候会是负数
      continue;
    }
    uint64_t num_free_entries = min(sq_wqe_cnt, cq_cnt) - num_active_sq_entries;//SQ上剩余可用条目数
    uint64_t num_entries_until_wave_last_entry = wave_sq_counter + num_active_lanes - db_touched; // 总共要post的 - touched;本 wavefront 最后一个条目距离 doorbell 的距离
    if (num_free_entries > num_entries_until_wave_last_entry) {//num_free_entries少1，应该num_free_entries+1 >= num_entries_until_wave_last_entry,更严格了
      break;
    }
    quiet();
  }

  outstanding_wqes[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
  //这里每个wqe都会新填写字段
  SegmentBuilder seg_build(my_sq_index, sq_buf); // wqe_idx和 base；取sq_buf对应my_sq_index的那段
  seg_build.update_ctrl_seg(my_sq_counter, opcode, 0, qp_num, MLX5_WQE_CTRL_CQ_UPDATE, 3, 0, 0);
  seg_build.update_raddr_seg(raddr, rkey);
  seg_build.update_data_seg(laddr, size, lkey);
  __atomic_signal_fence(__ATOMIC_SEQ_CST); // 与 __atomic_thread_fence 不同，__atomic_signal_fence 只在编译器层面生效，不会插入硬件指令。告诉编译器，不能把 fence 前后的内存操作重排，必须保持原有顺序。__ATOMIC_SEQ_CST强内存序

  if (is_leader) {//这里没有显式barrier，因为warp是lock step的
    uint64_t db_touched {0};
    do {
      db_touched = __hip_atomic_load(&sq_db_touched, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != wave_sq_counter);//确保之前的已经正确提交到硬件，这里保证了ring DB的次序，同时有多个warp要post的时候，ring DB是按WQE的顺序的，先提出post的先touch DB，同样下面444行用store也能保证sq_db_touched递增

    uint8_t *base_ptr = reinterpret_cast<uint8_t*>(sq_buf);
    uint64_t* ctrl_wqe_8B_for_db = reinterpret_cast<uint64_t*>(&base_ptr[64 * ((wave_sq_counter + num_wqes - 1) % sq_wqe_cnt)]);//WQE64字节，这里是指向最后一个WQE
    ring_doorbell(*ctrl_wqe_8B_for_db, wave_sq_counter + num_wqes);

    __hip_atomic_fetch_add(&quiet_posted, num_wqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&sq_db_touched, wave_sq_counter + num_wqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}
#endif // !GPUIB_IONIC
#endif // !GPUIB_BNXT

#ifndef GPUIB_BNXT
#ifdef GPUIB_IONIC
__device__ uint64_t QueuePair::post_wqe_amo(int pe, int32_t size, uintptr_t *raddr, uint8_t opcode,
                                            int64_t atomic_data, int64_t atomic_cmp, bool fetching) {
  uint64_t activemask = get_same_qp_lane_mask();
  uint32_t num_wqes = get_active_lane_count(activemask);
  uint32_t my_logical_lane_id = get_active_lane_num(activemask);
  bool is_leader{my_logical_lane_id == 0};
  const uint64_t leader_phys_lane_id = get_first_active_lane_id(activemask);
  uint32_t my_sq_prod = reserve_sq(activemask, num_wqes);
  uint32_t my_sq_pos = my_sq_prod + my_logical_lane_id;
  struct ionic_v1_wqe *wqe = &sq_buf[my_sq_pos & sq_mask];
  uint32_t cons;

  uint64_t* wave_fetch_atomic{nullptr};
  if (fetching) {
    if (is_leader) {
      auto res = fetching_atomic_freelist->pop_front();
      while (!res.success) {
        res = fetching_atomic_freelist->pop_front();
      }
      wave_fetch_atomic = res.value;
    }
    wave_fetch_atomic = (uint64_t*)__shfl((uint64_t)wave_fetch_atomic, leader_phys_lane_id);
  }

  wqe->base.wqe_id = my_sq_pos;
  wqe->base.op = opcode;
  wqe->base.num_sge_key = 1;
  wqe->base.flags = swap_endian_val<uint16_t>(0);
  wqe->base.imm_data_key = swap_endian_val<uint32_t>(0);

  wqe->atomic_v2.remote_va_high = swap_endian_val<uint32_t>(reinterpret_cast<uint64_t>(raddr) >> 32);
  wqe->atomic_v2.remote_va_low = swap_endian_val<uint32_t>(reinterpret_cast<uint64_t>(raddr));
  wqe->atomic_v2.remote_rkey = swap_endian_val<uint32_t>(rkey);
  wqe->atomic_v2.swap_add_high = swap_endian_val<uint32_t>(atomic_data >> 32);
  wqe->atomic_v2.swap_add_low = swap_endian_val<uint32_t>(atomic_data);
  wqe->atomic_v2.compare_high = swap_endian_val<uint32_t>(atomic_cmp >> 32);
  wqe->atomic_v2.compare_low = swap_endian_val<uint32_t>(atomic_cmp);

  if (fetching) {
    wqe->atomic_v2.local_va = swap_endian_val<uint64_t>(reinterpret_cast<uint64_t>(wave_fetch_atomic + my_logical_lane_id));
    wqe->atomic_v2.lkey = swap_endian_val<uint32_t>(fetching_atomic_lkey);
  } else {
    wqe->atomic_v2.local_va = swap_endian_val<uint64_t>(reinterpret_cast<uint64_t>(nonfetching_atomic));
    wqe->atomic_v2.lkey = swap_endian_val<uint32_t>(nonfetching_atomic_lkey);
  }

  cons = commit_sq(is_last_active_lane(activemask), my_sq_prod, num_wqes, wqe);

  uint64_t ret{0};
  if (fetching) {
    quiet_internal(activemask, cons);
    ret = wave_fetch_atomic[my_logical_lane_id];
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    if (is_leader) {
      fetching_atomic_freelist->push_back(wave_fetch_atomic);
    }
  }
  return ret;
}
#else // !GPUIB_IONIC || !GPUIB_BNXT
__device__ uint64_t QueuePair::post_wqe_amo(int pe, int32_t size, uintptr_t *raddr, uint8_t opcode,
                                            int64_t atomic_data, int64_t atomic_cmp, bool fetching) {
  uint64_t activemask = get_active_lane_mask();
  uint8_t num_active_lanes = get_active_lane_count(activemask);
  uint8_t my_logical_lane_id = get_active_lane_num(activemask);
  bool is_leader{my_logical_lane_id == 0};
  const uint64_t leader_phys_lane_id = get_first_active_lane_id(activemask);
  uint8_t num_wqes{num_active_lanes};
  uint64_t wave_sq_counter{0};

  if (is_leader) {
    wave_sq_counter = __hip_atomic_fetch_add(&sq_posted, num_wqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  wave_sq_counter = __shfl(wave_sq_counter, leader_phys_lane_id);
  uint64_t my_sq_counter = wave_sq_counter + my_logical_lane_id;
  uint64_t my_sq_index = my_sq_counter % sq_wqe_cnt;

  while (true) {
    uint64_t db_touched = __hip_atomic_load(&sq_db_touched, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t sunk = __hip_atomic_load(&sq_sunk, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    int64_t num_active_sq_entries = db_touched - sunk;
    if (num_active_sq_entries < 0) {
      continue;
    }
    uint64_t num_free_entries = min(sq_wqe_cnt, cq_cnt) - num_active_sq_entries;
    uint64_t num_entries_until_wave_last_entry = wave_sq_counter + num_active_lanes - db_touched;
    if (num_free_entries > num_entries_until_wave_last_entry) {
      break;
    }
    quiet();
  }

  uint64_t* wave_fetch_atomic{nullptr};
  if (fetching) {
    if (is_leader) {
      uint64_t db_touched {0};
      do {
        db_touched = __hip_atomic_load(&sq_db_touched, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      } while (db_touched != wave_sq_counter);

      auto res = fetching_atomic_freelist->pop_front();
      while (!res.success) {
        res = fetching_atomic_freelist->pop_front();
      }
      wave_fetch_atomic = res.value;
    }
    wave_fetch_atomic = (uint64_t*)__shfl((uint64_t)wave_fetch_atomic, leader_phys_lane_id);
  }

  outstanding_wqes[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;

  SegmentBuilder seg_build(my_sq_index, sq_buf);
  seg_build.update_ctrl_seg(my_sq_counter, opcode, 0, qp_num, MLX5_WQE_CTRL_CQ_UPDATE, 4, 0, 0);
  seg_build.update_raddr_seg(raddr, rkey);
  seg_build.update_atomic_seg(atomic_data, atomic_cmp);
  if (fetching) {
    seg_build.update_data_seg(wave_fetch_atomic + my_logical_lane_id, 8, fetching_atomic_lkey);
  } else {
    seg_build.update_data_seg(nonfetching_atomic, 8, nonfetching_atomic_lkey);
  }
  __atomic_signal_fence(__ATOMIC_SEQ_CST);

  if (is_leader) {
    uint64_t db_touched {0};
    do {
      db_touched = __hip_atomic_load(&sq_db_touched, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != wave_sq_counter);

    uint8_t *base_ptr = reinterpret_cast<uint8_t*>(sq_buf);
    uint64_t* ctrl_wqe_8B_for_db = reinterpret_cast<uint64_t*>(&base_ptr[64 * ((wave_sq_counter + num_wqes - 1) % sq_wqe_cnt)]);
    ring_doorbell(*ctrl_wqe_8B_for_db, wave_sq_counter + num_wqes);

    __hip_atomic_fetch_add(&quiet_posted, num_wqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&sq_db_touched, wave_sq_counter + num_wqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }

  uint64_t ret{0};
  if (fetching) {
    quiet();
    ret = wave_fetch_atomic[my_logical_lane_id];
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    if (is_leader) {
      fetching_atomic_freelist->push_back(wave_fetch_atomic);
    }
  }
  return ret;
}
#endif // !GPUIB_IONIC
#endif // !GPUIB_BNXT

/******************************************************************************
 ****************************** SHMEM INTERFACE *******************************
 *****************************************************************************/
__device__ void QueuePair::put_nbi(void *dest, const void *source, size_t nelems, int pe) {
  uintptr_t *src = reinterpret_cast<uintptr_t*>(const_cast<void*>(source));
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  post_wqe_rma(pe, nelems, src, dst, GPUIB_OP_RDMA_WRITE);//
}

__device__ int64_t QueuePair::atomic_fetch(void *dest, int64_t atomic_data, int64_t atomic_cmp, int pe, uint8_t atomic_op) {
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  return post_wqe_amo(pe, sizeof(int64_t), dst, atomic_op, atomic_data, atomic_cmp, true);
}

__device__ void QueuePair::atomic_nofetch(void *dest, int64_t atomic_data, int64_t atomic_cmp, int pe, uint8_t atomic_op) {
  uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);
  post_wqe_amo(pe, sizeof(int64_t), dst, atomic_op, atomic_data, atomic_cmp, false);
}

}  // namespace rocshmem
