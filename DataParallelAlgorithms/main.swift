//
//  main.swift
//  DataParallelAlgorithms
//
//  Created by Eoin Roe on 17/09/2022.
//

import Metal

var device = MTLCreateSystemDefaultDevice()

guard device != nil else {
    print("Metal is not supported on this device")
    exit(EXIT_FAILURE)
}

// Create the custom object used to encapsulate the Metal code.
// Initializes objects to communicate with the GPU.
// var adder = MetalAdder(device: device!)

// Create the buffers to hold data.
// adder.prepareData()

// Send a command to the GPU to perform the calculation.
// adder.sendComputeCommand()

// Create the custom object used to encapsulate the Metal code.
// Initializes objects to communicate with the GPU.
var reduction = ParallelReduction(device: device!)

// Create the buffers to hold data.
reduction.prepareData()

// Send a command to the GPU to perform the calculation.
reduction.sendComputeCommand()
