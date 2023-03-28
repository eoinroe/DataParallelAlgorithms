//
//  MetalAdder.swift
//  DataParallelAlgorithms
//
//  Created by Eoin Roe on 17/09/2022.
//

import Metal

fileprivate let arrayLength = 1 << 4
fileprivate let bufferSize = arrayLength * MemoryLayout<Float>.size

class MetalAdder {
    var device: MTLDevice
    
    // The compute pipeline generated from the compute kernel in the .metal shader file.
    var computePipeline: MTLComputePipelineState!
    
    // The command queue used to pass commands to the device.
    var commandQueue: MTLCommandQueue!
    
    // Buffers to hold data.
    var bufferA: MTLBuffer!
    var bufferB: MTLBuffer!
    var bufferResult: MTLBuffer!
    
    init(device: MTLDevice) {
        self.device = device
        
        let defaultLibrary = device.makeDefaultLibrary()
        if defaultLibrary == nil {
            print("Failed to find the default library.")
        }
        
        guard let addFunction = defaultLibrary?.makeFunction(name: "add_arrays") else {
            fatalError("Failed to find the adder function.")
        }
        
        // Create a compute pipeline state object.
        do {
            computePipeline = try device.makeComputePipelineState(function: addFunction)
        } catch let error {
            print(error.localizedDescription)
        }
        
        commandQueue = device.makeCommandQueue()
    }
    
    func prepareData() {
        // Allocate three buffers to hold our initial data and the result.
        bufferA = device.makeBuffer(length: bufferSize)
        bufferB = device.makeBuffer(length: bufferSize)
        bufferResult = device.makeBuffer(length: bufferSize)
        
        generateRandomFloatData(buffer: bufferA)
        generateRandomFloatData(buffer: bufferB)
    }
    
    func generateRandomFloatData(buffer: MTLBuffer) {
        let dataPtr = buffer.contents().assumingMemoryBound(to: Float.self)
        
        for i in 0..<arrayLength {
            dataPtr[i] = Float.random(in: 0...1)
        }
    }
    
    /*
    func inspectRandomFloatData(buffer: MTLBuffer) {
        let dataPtr = buffer.contents().assumingMemoryBound(to: Float.self)
        
        for i in 0..<arrayLength {
            let x = dataPtr[i]
            print(x)
        }
    }
     */
    
    func sendComputeCommand() {
        // Create command buffer to hold commands.
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                
                encodeAddCommand(computeEncoder: computeEncoder)
                
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
    
    func encodeAddCommand(computeEncoder: MTLComputeCommandEncoder) {
        // Encode the pipeline state object and its parameters.
        computeEncoder.setComputePipelineState(computePipeline)
        
        computeEncoder.setBuffer(bufferA,      offset: 0, index: 0)
        computeEncoder.setBuffer(bufferB,      offset: 0, index: 1)
        computeEncoder.setBuffer(bufferResult, offset: 0, index: 2)
        
        
        // The add_arrays functions uses a 1D array, so the sample creates a 1D grid of
        // size (dataSize x 1 x 1), from which Metal generates indices between 0 and dataSize-1.
        let gridSize = MTLSizeMake(arrayLength, 1, 1)
        
        // The app asks the pipeline state object for the largest possible threadgroup...
        var threadGroupSize = computePipeline.maxTotalThreadsPerThreadgroup
        
        // ... and shrinks it if that size is larger than the size of the data set.
        if threadGroupSize > arrayLength {
            threadGroupSize = arrayLength
        }
        
        // Metal subdivides the grid into smaller grids called threadgroups.
        // Each threadgroup is calculated separately.
        let threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1)
        
        computeEncoder.dispatchThreads(gridSize,
                                       threadsPerThreadgroup: threadgroupSize)
    }
    
    func verifyResults() {
        let a = bufferA.contents().assumingMemoryBound(to: Float.self)
        let b = bufferB.contents().assumingMemoryBound(to: Float.self)
        let result = bufferResult.contents().assumingMemoryBound(to: Float.self)
        
        for index in 0..<arrayLength {
            if result[index] != (a[index] + b[index]) {
                print("Compute ERROR.")
            }
        }
        print("Compute results as expected.")
    }
}
