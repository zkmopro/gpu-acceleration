//
//  MSMBenchmarkView.swift
//  GPU Acceleration
//
//  Created by Fuchuan Chung on 2024/4/21.
//  Copyright © 2024 CocoaPods. All rights reserved.
//

import SwiftUI
import moproFFI

struct AlgorithmBenchmark: Identifiable {
    let id = UUID()
    var algorithm: String
    var avgMsmTime: Double
    var diffWithBaseline: Double
}

struct MSMBenchmarkView: View {
    @State private var selectedAlgorithms: Set<String> = ["Arkwork (Baseline)", "Metal (GPU)"]
    let algorithms = ["Arkwork (Baseline)", "Metal (GPU)", "Bucket Wise Msm", "Precompute Msm"]
    @State private var benchmarkResults: [AlgorithmBenchmark] = []
    @State private var isSubmitting: Bool = false
    
    let msmBenchmarkMapping: [String: (UInt32, UInt32, String) throws -> BenchmarkResult] = [
        "Arkwork (Baseline)": arkworksPippenger,
        "Metal (GPU)": metalMsm,
        "Bucket Wise Msm": bucketWiseMsm,
        "Precompute Msm": precomputeMsm
    ]

    var body: some View {
        ZStack {
            Color.black.edgesIgnoringSafeArea(.all)  // Set the background to gray
            
            ScrollView {
                VStack(spacing: 20) {
                    Text("MSM Benchmark")
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                    
                    VStack(alignment: .leading, spacing: 10) {
                        Text("SELECT MSM ALGORITHMS")
                            .font(.subheadline)
                            .foregroundColor(.white)
                        
                        VStack(spacing: 0) {
                            ForEach(algorithms, id: \.self) { algorithm in
                                Button(action: {
                                    toggleAlgorithm(algorithm)
                                }) {
                                    HStack {
                                        Text(algorithm)
                                        Spacer()
                                        if selectedAlgorithms.contains(algorithm) {
                                            Image(systemName: "checkmark")
                                        }
                                    }
                                    .padding()
                                    .background(selectedAlgorithms.contains(algorithm) ? Color.blue.opacity(0.3) : Color.gray.opacity(0.3))
                                    .foregroundColor(.white)
                                }
                                .buttonStyle(PlainButtonStyle())
                                Divider().background(Color.white.opacity(0.3))
                            }
                        }
                        .background(Color.gray.opacity(0.2))
                        .cornerRadius(10)
                    }
                    
                    VStack(alignment: .leading, spacing: 10) {
                        Text("BENCHMARK RESULTS")
                            .font(.subheadline)
                            .foregroundColor(.white)
                        
                        VStack(spacing: 0) {
                            HStack {
                                Text("Algorithm")
                                Spacer()
                                Text("Avg Time\n(ms)")
                                Text("Diff(%)")
                                    .frame(width: 70, alignment: .trailing)
                            }
                            .padding()
                            .background(Color.gray.opacity(0.3))
                            .font(.subheadline)
                            .foregroundColor(.white)
                            
                            if benchmarkResults.isEmpty {
                                Text("No results yet")
                                    .padding()
                                    .frame(maxWidth: .infinity, alignment: .center)
                                    .foregroundColor(.white)
                            } else {
                                ForEach(benchmarkResults) { result in
                                    HStack {
                                        Text(result.algorithm)
                                        Spacer()
                                        Text(String(format: "%.2f", result.avgMsmTime))
                                            .frame(width: 80, alignment: .trailing)
                                        Text(String(format: "%.2f", result.diffWithBaseline))
                                            .frame(width: 70, alignment: .trailing)
                                    }
                                    .padding()
                                    .foregroundColor(.white)
                                    if result.id != benchmarkResults.last?.id {
                                        Divider().background(Color.white.opacity(0.3))
                                    }
                                }
                            }
                        }
                        .background(Color.gray.opacity(0.2))
                        .cornerRadius(10)
                    }
                    
                    Spacer()
                    
                    Button(action: submitAction) {
                        Text("Generate Benchmarks")
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                    .disabled(isSubmitting || selectedAlgorithms.isEmpty)
                }
                .padding()
            }
        }
    }

    func toggleAlgorithm(_ algorithm: String) {
        if selectedAlgorithms.contains(algorithm) {
            selectedAlgorithms.remove(algorithm)
        } else {
            selectedAlgorithms.insert(algorithm)
        }
    }

    func submitAction() {
        let instanceSize: UInt32 = 10
        let numInstance: UInt32 = 5
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.path
        
        isSubmitting = true
        benchmarkResults.removeAll()
        
        DispatchQueue.global(qos: .userInitiated).async {
            var tempResults: [AlgorithmBenchmark] = []
            var baselineTiming: Double = 0.0
            
            for algorithm in selectedAlgorithms.sorted() {
                if let benchmarkFunction = msmBenchmarkMapping[algorithm] {
                    do {
                        let benchData = try benchmarkFunction(instanceSize, numInstance, documentsPath)
                        
                        if algorithm == "Arkwork (Baseline)" {
                            baselineTiming = benchData.avgProcessingTime
                        }
                        
                        let diff = algorithm == "Arkwork (Baseline)" ? 0.0 :
                            (baselineTiming - benchData.avgProcessingTime) / baselineTiming * 100
                        
                        let algorithmBenchmark = AlgorithmBenchmark(
                            algorithm: algorithm,
                            avgMsmTime: benchData.avgProcessingTime,
                            diffWithBaseline: diff
                        )
                        tempResults.append(algorithmBenchmark)
                    } catch {
                        print("Error running benchmark for \(algorithm): \(error)")
                    }
                }
            }

            DispatchQueue.main.async {
                benchmarkResults = tempResults
                isSubmitting = false
            }
        }
    }
}
