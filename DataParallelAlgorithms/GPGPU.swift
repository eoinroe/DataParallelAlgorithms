//
//  GPGPU.swift
//  DataParallelAlgorithms
//
//  Created by Eoin Roe on 19/09/2022.
//

import Metal

protocol GPGPU {
    
    func prepareData()
    
    func sendComputeCommand()
    
    func verifyResults()
    
}
