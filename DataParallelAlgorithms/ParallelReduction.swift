//
//  ParallelReduction.swift
//  DataParallelAlgorithms
//
//  Created by Eoin Roe on 17/09/2022.
//

import Metal

class ParallelReduction: GPGPU {
    var device: MTLDevice
    
    // The compute pipelines generated from the compute kernels in the .metal shader file.
    var interleavedAddressingPipelineState: MTLComputePipelineState!
    var sequentialAddressingPipelineState: MTLComputePipelineState!
    var kernelDecompositionPipelineState: MTLComputePipelineState!
    var simdgroupSumPipelineState: MTLComputePipelineState!
    
    // The command queue used to pass commands to the device.
    var commandQueue: MTLCommandQueue!
    
    // Buffers to hold data.
    var bufferA: MTLBuffer!
    // var bufferB: MTLBuffer!
    
    var N: Int = 2048
    
    // In order to simplify the implementation we will only
    // consider arrays with sizes that equal to powers of 2.
    // var inputArray = Array(repeating: Float(1), count: 16)
    
    var inputArray: [Float] = []
    
    // As long as we do not need to keep the original GPU array,
    // we can perform the reduction in-place directly on the input
    // array placed in the global memory.
    
    var totalSum: MTLBuffer!
    
    init(device: MTLDevice) {
        self.device = device
        
        let defaultLibrary = device.makeDefaultLibrary()
        if defaultLibrary == nil {
            print("Failed to find the default library.")
        }
        
        guard let interleavedAddressingFunction = defaultLibrary?.makeFunction(name: "interleaved_addressing") else {
            fatalError("Failed to find the kernel function.")
        }
        
        // Create a compute pipeline state object.
        do {
            interleavedAddressingPipelineState = try device.makeComputePipelineState(function: interleavedAddressingFunction)
        } catch let error {
            print(error.localizedDescription)
        }
        
        guard let sequentialAddressingFunction = defaultLibrary?.makeFunction(name: "sequential_addressing") else {
            fatalError("Failed to find the kernel function.")
        }
        
        do {
            sequentialAddressingPipelineState = try device.makeComputePipelineState(function: sequentialAddressingFunction)
        } catch let error {
            print(error.localizedDescription)
        }
        
        guard let kernelDecompositionFunction = defaultLibrary?.makeFunction(name: "kernel_decomposition") else {
            fatalError("Failed to find the kernel function.")
        }
        
        do {
            kernelDecompositionPipelineState = try device.makeComputePipelineState(function: kernelDecompositionFunction)
        } catch let error {
            print(error.localizedDescription)
        }
        
        guard let simdgroupSumFunction = defaultLibrary?.makeFunction(name: "reduce_sum") else {
            fatalError("Failed to find the kernel function.")
        }
        
        do {
            simdgroupSumPipelineState = try device.makeComputePipelineState(function: simdgroupSumFunction)
        } catch let error {
            print(error.localizedDescription)
        }
        
        commandQueue = device.makeCommandQueue()
    }
    
    func prepareData() {
        generateRandomFloatData()
        bufferA = device.makeBuffer(bytes: inputArray, length: MemoryLayout<Float>.size * inputArray.count)
        print("Buffer size: ", MemoryLayout<Float>.size * inputArray.count)
        
        totalSum = device.makeBuffer(length: MemoryLayout<Float>.stride)
    }
    
    func generateRandomFloatData() {
        for _ in 0..<N {
            // inputArray.append(Float.random(in: 0...1))
            
            // For testing...
            inputArray.append(Float(Int.random(in: 0...5)))
        }
    }
    
    func sendComputeCommand() {
        // Create command buffer to hold commands.
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                
                // interleavedAddressing(computeEncoder: computeEncoder)
                // sequentialAddressing(computeEncoder: computeEncoder)
                // kernelDecomposition(computeEncoder: computeEncoder)
                simdgroupSum(computeEncoder: computeEncoder)
                
                // End the compute pass.
                computeEncoder.endEncoding()
                
            }
            
            // Execute the command.
            commandBuffer.commit()
            
            // Normally, you want to do other work in your app while the GPU is running,
            // but in this example, the code simply blocks until the calculation is complete.
            commandBuffer.waitUntilCompleted()
        }
        
        verifyResults()
    }
    
    /// On the high-level, this function should contain a loop that always computes parameters for
    /// launching the kernel, sets up correct arguments, and enqueues the kernel in the command queue.
    func interleavedAddressing(computeEncoder: MTLComputeCommandEncoder) {
        /**
         Notice that the result from one step is always used as the input for the next step.
         This means that we need to synchronize work-items... otherwise work-items from
         one group could perform next step before the current step is finished
         by other work groups...
         
         ... we can exploit the fact that kernels are executed in a non-overlapping consecutive order:
         next kernel can start only after the current one is finished. Therefore, we can split the reduction
         into a number of kernel calls.
         */
        
        var numThreads = inputArray.count
        let numberOfSteps = log2(Float(inputArray.count))
        
        for step in 0..<Int(numberOfSteps) {
            numThreads /= 2
            print("Num threads: ", numThreads)
            
            let stride = UInt32(truncating: pow(2, step) as NSNumber)
            print("Stride: ", stride)
            
            let offset = stride * 2
            print("Offset: ", offset)
            
            // Encode the pipeline state object and its parameters.
            computeEncoder.setComputePipelineState(interleavedAddressingPipelineState)
            
            computeEncoder.setBuffer(bufferA, offset: 0, index: 0)
            
            computeEncoder.setValue(stride, at: 1)
            computeEncoder.setValue(offset, at: 2)
            
            let gridSize = MTLSizeMake(numThreads, 1, 1)
            
            // The app asks the pipeline state object for the largest possible threadgroup...
            var threadGroupSize = interleavedAddressingPipelineState.maxTotalThreadsPerThreadgroup
            
            // ... and shrinks it if that size is larger than the size of the data set.
            if threadGroupSize > numThreads {
                threadGroupSize = numThreads
            }
            
            // Metal subdivides the grid into smaller grids called threadgroups.
            // Each threadgroup is calculated separately.
            let threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1)
            
            computeEncoder.dispatchThreads(gridSize,
                                           threadsPerThreadgroup: threadgroupSize)
        }
        
    }
    
    func sequentialAddressing(computeEncoder: MTLComputeCommandEncoder) {
        /**
         The biggest drawback of the interleaved addressing is that the accesses
         to the global memory are not fully coalesced. The coalescing can be achieved
         quite easily just by using different - sequential - addressing. In each step,
         the threads should read two consecutive chunks of memory (first and second half of the elements), reduce them, and write the results back again in a coalesced manner.
         */
        var numThreads = inputArray.count
        let numberOfSteps = log2(Float(inputArray.count))
        
        for _ in 0..<Int(numberOfSteps) {
            numThreads /= 2
            print("Num threads: ", numThreads)
            
            // Encode the pipeline state object and its parameters.
            computeEncoder.setComputePipelineState(sequentialAddressingPipelineState)
            
            computeEncoder.setBuffer(bufferA, offset: 0, index: 0)
            
            computeEncoder.setValue(numThreads, at: 1)
            
            let gridSize = MTLSizeMake(numThreads, 1, 1)
            
            // The app asks the pipeline state object for the largest possible threadgroup...
            var threadGroupSize = sequentialAddressingPipelineState.maxTotalThreadsPerThreadgroup
            
            // ... and shrinks it if that size is larger than the size of the data set.
            if threadGroupSize > numThreads {
                threadGroupSize = numThreads
            }
            
            // Metal subdivides the grid into smaller grids called threadgroups.
            // Each threadgroup is calculated separately.
            let threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1)
            
            computeEncoder.dispatchThreads(gridSize,
                                           threadsPerThreadgroup: threadgroupSize)
        }
    }
    
    func kernelDecomposition(computeEncoder: MTLComputeCommandEncoder) {
        /**
         Use 256 threads (work-items) for 512 elements.
         */
        // Encode the pipeline state object and its parameters.
        computeEncoder.setComputePipelineState(kernelDecompositionPipelineState)
        
        computeEncoder.setBuffer(bufferA, offset: 0, index: 0)
        
        computeEncoder.setThreadgroupMemoryLength(MemoryLayout<Float>.stride * 32, index: 0)
        
        let gridSize = MTLSizeMake(256, 1, 1)
        
        // The app asks the pipeline state object for the largest possible threadgroup...
        // var threadGroupSize = sequentialAddressingPipelineState.maxTotalThreadsPerThreadgroup
        
        // ... and shrinks it if that size is larger than the size of the data set.
        // if threadGroupSize > numThreads {
        //     threadGroupSize = numThreads
        // }
        
        // Metal subdivides the grid into smaller grids called threadgroups.
        // Each threadgroup is calculated separately.
        let threadgroupSize = MTLSizeMake(32, 1, 1)
        
        computeEncoder.dispatchThreads(gridSize,
                                       threadsPerThreadgroup: threadgroupSize)
    }
    
    func simdgroupSum(computeEncoder: MTLComputeCommandEncoder) {
        // Encode the pipeline state object and its parameters.
        computeEncoder.setComputePipelineState(simdgroupSumPipelineState)
        
        computeEncoder.setBuffer(bufferA,  offset: 0, index: 0)
        computeEncoder.setBuffer(totalSum, offset: 0, index: 1)
        
        let simdSize = simdgroupSumPipelineState.threadExecutionWidth
        print("Simd size: ", simdSize)
        
        computeEncoder.setThreadgroupMemoryLength(MemoryLayout<Float>.stride * simdSize, index: 0)
        
        // The app asks the pipeline state object for the largest possible threadgroup...
        let maxTotalThreadsPerThreadgroup = simdgroupSumPipelineState.maxTotalThreadsPerThreadgroup
        print("Max total threads per threadgroup: ", maxTotalThreadsPerThreadgroup)
        
        let gridSize = MTLSizeMake(N, 1, 1)
        let threadgroupSize = MTLSizeMake(maxTotalThreadsPerThreadgroup, 1, 1)
        
        computeEncoder.dispatchThreads(gridSize,
                                       threadsPerThreadgroup: threadgroupSize)
        
    }
    
    func verifyResultsKernelDecomposition() {
        let a = bufferA.contents().assumingMemoryBound(to: Float.self)
        
        var result: Float = 0
        
        for i in 0..<8 {
            result += a[i]
        }
        
        result = round(result * 100) / 100
        
        var total = inputArray.reduce(0, +)
        total = round(total * 100) / 100
        
        if result != total {
            print("Compute Error.")
        } else {
            print("Compute results as expected.")
        }
        
        print(result)
        print(total)
    }
    
    func verifyResults() {
        // let a = bufferA.contents().assumingMemoryBound(to: Float.self)
        let a = totalSum.contents().assumingMemoryBound(to: Float.self)
        let result = round(a[0] * 100) / 100
        
        var total = inputArray.reduce(0, +)
        total = round(total * 100) / 100
        
        if result != total {
            print("Compute Error.")
        } else {
            print("Compute results as expected.")
        }
        
        print(result)
        print(total)
    }
    
    
}
