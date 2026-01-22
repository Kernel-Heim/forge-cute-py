import operator
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32
from typing import Callable

class SafeSoftmax:
    """Safe softmax kernel operation using CuTe DSL."""

    def __init__(self, dtype: type, N=1024, threads_per_row=256):
        self.dtype = dtype
        self.N = N
        self.threads_per_row = threads_per_row
        self.rows_per_block = 1024 // threads_per_row
        self.warps_per_row = threads_per_row // cute.arch.WARP_SIZE

    @cute.jit
    def _row_reduce(self, val: cute.Numeric, op: Callable, init_val: cute.Numeric):
        """
        Performs a reduction across a row using `threads_per_row` threads (`warps_per_row` warps).
        """
        # Warp Reduction: 32 threads -> 1 val each warp, ttl 8 vals
        val = cute.arch.warp_reduction(val, op, threads_in_group=cute.arch.WARP_SIZE)

        # Block Reduction: 8 warps per row -> 8 vals in smem as reduction buffer
        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_layout(
            (self.warps_per_row, self.rows_per_block), 
            stride=(1, self.warps_per_row) # Column-major
        )
        reduction_buffer = smem.allocate_tensor(Float32, smem_layout)

        warp_idx_global = cute.arch.warp_idx()
        row_in_block    = warp_idx_global // self.warps_per_row
        warp_in_row     = warp_idx_global % self.warps_per_row
        lane_idx        = cute.arch.lane_idx()

        # Write to Shared Memory
        # Only the first thread of each warp writes the result
        if lane_idx == 0:
            reduction_buffer[warp_in_row, row_in_block] = val
        
        cute.arch.barrier()

        # Read back
        final_val = val

        # Only the first warp of each row gathers the partials        
        # Final Reduction
        if warp_in_row == 0:
            # Load the 8 partial results from smem
            reduce_val = init_val
            if lane_idx < self.warps_per_row:
                reduce_val = reduction_buffer[lane_idx, row_in_block]
            # Reduce these 8 values in single warp
            final_val = cute.arch.warp_reduction(reduce_val, op, threads_in_group=self.warps_per_row)
            # Write final result back to smem so all warps in this row
            if lane_idx == 0:
                reduction_buffer[0, row_in_block] = final_val

        cute.arch.barrier()
        
        return reduction_buffer[0, row_in_block]

    @cute.jit
    def __call__(self, input: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream = None):
        M, N = input.shape
        
        # Block Dimensions: (256, 4, 1)
        block_dim = (self.threads_per_row, self.rows_per_block, 1)
        
        # Grid Dimensions: M / 4
        grid_dim = (cute.ceil_div(M, self.rows_per_block), 1, 1)

        # Vectorize by 4*float32 -> 128 bit loads)
        vectorized_input = cute.zipped_divide(input, (1, 4))
        vectorized_output = cute.zipped_divide(output, (1, 4))

        self.kernel(vectorized_input, vectorized_output).launch(
            grid=grid_dim,
            block=block_dim,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, input: cute.Tensor, output: cute.Tensor):
        """
        Safe Softmax kernel CuTe DSL implementation.

        Strategy:
        0. Vectorized load
        1. Find max value of the row
            1. Thread reduction (read and write to registers):
                Find max value of each slice/sub-tensor.
            2. Warp reduction (read and write to registers)
                Find max value within each wrap.
                Store the max values in shared memory.
            3. Block reduction (read and write to shared memory)
                Find global max value using max values stored in shared memory of the thread block.
        2. Safe exponentials (Element-wise)
            1. Subtract max from each element in the slice.
            2. Compute exponential of each element in the slice.
        3. Sum computed values of the row
            1. Thread reduction (read and write to registers):
                Add up values in each slice/sub-tensor.
            2. Warp reduction (read and write to registers)
                Add up values within each wrap.
                Store the summations in shared memory.
            3. Block reduction (read and write to shared memory)
                Get global summation using sum values stored in shared memory of the thread block.
        4. Broadcast and Normalize (Element-wise)
        5. Vectorized store
        """

        # tidx maps to columns (vectors), tidy maps to rows
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # Global Row Index
        mi = bidx * self.rows_per_block + tidy
        # Column Index (in slice-unit)
        ni = tidx

        # 0. Read
        vals = input[(None, (mi, ni))].load()
        # 1. Max
        # Reduce from local vector (each sub-tensor has 4 elements)
        local_max = vals.reduce(cute.ReductionOp.MAX, init_val=float('-inf'), reduction_profile=0)
        # Reduce across the whole row
        global_max = self._row_reduce(local_max, cute.arch.fmax, float('-inf'))
        # 2. Exp
        safe_vals = vals - global_max
        exp_vals = cute.exp(safe_vals)
        # 3. Sum
        # Reduce from local vector (each sub-tensor has 4 elements)
        local_sum = exp_vals.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
        # Reduce across the whole row
        global_sum = self._row_reduce(local_sum, operator.add, 0.0)
        # 4. Normalize
        softmax_vals = exp_vals / global_sum
        # 5. Write
        output[(None, (mi, ni))] = softmax_vals