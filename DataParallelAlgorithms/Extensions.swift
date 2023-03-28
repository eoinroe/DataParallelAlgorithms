//
//  Extensions.swift
//  DataParallelAlgorithms
//
//  Created by Eoin Roe on 19/09/2022.
//

import MetalKit

public extension MTLComputeCommandEncoder {
    func setValue<T>(_ value: T, at index: Int) {
        var t = value
        self.setBytes(&t, length: MemoryLayout<T>.stride, index: index)
    }
}
