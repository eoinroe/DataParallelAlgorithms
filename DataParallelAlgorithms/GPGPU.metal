//
//  GPGPU.metal
//  DataParallelAlgorithms
//
//  Created by Eoin Roe on 17/09/2022.
//

#include <metal_stdlib>
using namespace metal;

/// This is a Metal Shading Language (MSL) function equivalent to the add_arrays() C function, used to perform the calculation on a GPU.
void kernel add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    // The for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] + inB[index];
}

kernel void interleaved_addressing(device float* numbers,
                                   constant uint& stride,
                                   constant uint& offset,
                                   uint gid [[thread_position_in_grid]])
{
    uint index = gid * offset;
    numbers[index] = numbers[index] + numbers[index + stride];
}

kernel void sequential_addressing(device float* numbers,
                                  constant int& offset,
                                  uint gid [[thread_position_in_grid]])
{
    // gid is short for global index
    numbers[gid] = numbers[gid] + numbers[gid + offset];
}

kernel void kernel_decomposition(device float* numbers,
                                 threadgroup float* local_memory,
                                 uint threadsPerThreadgroup [[threads_per_threadgroup]],
                                 uint tid [[threadgroup_position_in_grid]],
                                 uint lid [[thread_position_in_threadgroup]],
                                 uint gid [[thread_position_in_grid]])
{
    // You can improve the performance by computing the first
    // reduction step before storing the data in the local memory.
    local_memory[lid] = numbers[gid] + numbers[gid + 256];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // N = 32
    uint offset = threadsPerThreadgroup / 2;
    
    // Local reduction of the 256 elements in the local memory is achieved by a single for loop.
    for (int i = 0; i < 5; i++) {
        // Each iteration should use only the appropriate number of threads
        // (use an if statement and thread ID to enable the computation only for some threads)
        if (lid < offset) {
            local_memory[lid] = local_memory[lid] + local_memory[lid + offset];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        offset /= 2;
    }
    
    numbers[tid] = local_memory[0];
    
    /*
    
    threadgroup_barrier(mem_flags::mem_device);
    
    offset = 8 / 2;
    
    for (int i = 0; i < 3; i++) {
        if (gid < offset) {
            numbers[gid] = numbers[gid] + numbers[gid + offset];
        }
        
        threadgroup_barrier(mem_flags::mem_device);
        
        offset /= 2;
    }
     
     */
    
    /*
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Ensure correct ordering of memory operations to device memory.
    // threadgroup_barrier(mem_flags::mem_device);
    
    // Write the result back to the global device memory.
    if (gid == 0) {
        float result = numbers[0];
    
        for (int i = 1; i < 8; i++) {
            result += numbers[i];
        }
    
        numbers[gid] = result;
    }
     
     */
}


// Load the data into local memory.
// numbers[lid] = numbers[gid];

// -*- Perform the reduction -*-

// https://developer.apple.com/videos/play/tech-talks/10858
kernel void reduce_sum(device int* input_array,
                       device atomic_int* total_sum,
                       threadgroup int* simdSumArray,
                       // The scalar index of a SIMD-group within a threadgroup.
                       uint simd_group_id  [[simdgroup_index_in_threadgroup]],
                       // The scalar index of a thread within a SIMD-group.
                       uint simd_lane_id   [[thread_index_in_simdgroup]],
                       uint lid            [[thread_position_in_threadgroup]],
                       uint gid    [[thread_position_in_grid]])
{
    int a = input_array[gid];
    
    int simdgroup_sum = simd_sum(a);
    simdSumArray[simd_group_id] = simdgroup_sum;
    
    threadgroup int sum = 0;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Also, works without this if statement since every simdgroup
    // in the threadgroup will perform the exact same calculation.
    if (simd_group_id == 0) {
        // simd_lane_id will be a number between 0 - 31
        int b = simdSumArray[simd_lane_id];
        sum = simd_sum(b);
    }
    
    if (lid == 0)
        atomic_fetch_add_explicit(total_sum, sum, memory_order_relaxed);
}


// kernel void reduce_sum(device float* input_array,
//                        device atomic_float* total_sum,
//                        threadgroup float* simdSumArray,
//                        // The scalar index of a SIMD-group within a threadgroup.
//                        uint simd_group_id  [[simdgroup_index_in_threadgroup]],
//                        // The number of SIMD-groups in a threadgroup.
//                        uint num_groups     [[simdgroups_per_threadgroup]],
//                        // The scalar index of a thread within a SIMD-group.
//                        uint simd_lane_id   [[thread_index_in_simdgroup]],
//                        //
//                        uint simd_size      [[threads_per_simdgroup]],
//                        uint lid            [[thread_position_in_threadgroup]],
//                        uint read_offset    [[thread_position_in_grid]])
// {
//     threadgroup float SSA[32];
//
//     float a = input_array[read_offset];
//
//     float simdgroup_sum = simd_sum(a);
//     SSA[simd_group_id] = simdgroup_sum;
//
//     threadgroup float sum = 0;
//
//     threadgroup_barrier(mem_flags::mem_threadgroup);
//
//     // Also, works without this if statement since every simdgroup
//     // in the threadgroup will perform the exact same calculation.
//     if (simd_group_id == 0) {
//         // simd_lane_id will be a number between 0 - 31
//         float b = SSA[simd_lane_id];
//         sum = simd_sum(b);
//     }
//
//     if (lid == 0)
//         atomic_fetch_add_explicit(total_sum, sum, memory_order_relaxed);
// }
