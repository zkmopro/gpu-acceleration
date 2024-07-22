//
//  MSMBenchmarkView.swift
//  GPU Acceleration
//
//  Created by Fuchuan Chung on 2024/4/21.
//  Copyright © 2024 CocoaPods. All rights reserved.
//

import Foundation
import SwiftUI
import moproFFI


struct AlgorithmBenchmark {
    var algorithm: String
    var avgMsmTime: Double
    var diffWithBaseline: Double // Baseline is Arkwork Vanilla MSM
}

struct MSMBenchmarkView: View {
    @State private var selectedAlgorithms: Set<Int> = [0] // Default to select the baseline MSM algorithm
    let algorithms = ["Arkwork (Baseline)", "Metal (GPU)"]
    @State private var benchmarkResults: [AlgorithmBenchmark] = []
    @State private var isSubmitting: Bool = false
    
    // setting up msm algorithm mapping
    let msmBenchmarkMapping:
    [ String: (
        UInt32,
        UInt32,
        String
    ) throws -> BenchmarkResult] = [
        "Arkwork (Baseline)": arkworksPippenger,
        "Metal (GPU)": metalMsm,
        // "TrapdoorTech Zprize": trapdoortechZprizeMsm,
    ]

    var body: some View {
        NavigationView {
            VStack {
                // The MSM algorithm Lists
                List {
                    Section(header: Text("Select MSM Algorithms")) {
                        ForEach(algorithms.indices, id: \.self) { index in
                            HStack {
                                Text("\(index + 1). \(algorithms[index])")
                                Spacer()
                                if selectedAlgorithms.contains(index) {
                                    Image(systemName: "checkmark")
                                }
                            }
                            .onTapGesture {
                                // select the algorithms
                                if selectedAlgorithms.contains(index) {
                                    selectedAlgorithms.remove(index)
                                } else {
                                    selectedAlgorithms.insert(index)
                                }
                            }
                            .foregroundColor(.black)
                            .listRowBackground(Color.white)
                        }
                    }

                    // result lists
                    Section(header: Text("Benchmark Results")) {
                        // Adding titles to the table-like structure
                        HStack {
                            Text("Algorithm")
                                .bold()
                                .frame(width: 120, alignment: .leading)
                            Spacer()
                            Text("Avg Time (ms)")
                                .bold()
                                .frame(width: 120, alignment: .trailing)
                            Text("Diff(%)")
                                .bold()
                                .frame(width: 80, alignment: .trailing)
                        }
                        .foregroundColor(.white)
                        .listRowBackground(Color.gray)
                        
                        // List of results
                        ForEach(benchmarkResults, id: \.algorithm) { result in
                            HStack {
                                Text(result.algorithm)
                                    .frame(width: 120, alignment: .leading)
                                Spacer()
                                Text("\(String(format: "%.2f", result.avgMsmTime))")
                                    .frame(width: 120, alignment: .trailing)
                                Text("\(String(format: result.diffWithBaseline > 0 ? "+%.2f" : "%.2f", result.diffWithBaseline))")
                                    .frame(width: 80, alignment: .trailing)
                            }
                            .foregroundColor(.black)
                            .listRowBackground(Color.white)
                        }
                    }
                }
                .listStyle(DefaultListStyle())
                .background(Color.black.edgesIgnoringSafeArea(.all))

                Button("Generate Benchmarks") {
                    submitAction()
                }
                
                .padding()
                .background(isSubmitting ? Color.gray : Color.blue)
                .foregroundColor(.white)
                .cornerRadius(5)
                .disabled(isSubmitting)
            }
            .navigationBarTitle("MSM Benchmark", displayMode: .inline)
            .navigationBarHidden(false)
        }
    }

    func submitAction() {
        let instanceSize: UInt32 = 10;
        let numInstance: UInt32 = 5;
        let documentsUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let documentsPath = documentsUrl.path
        
        isSubmitting = true
        
        // File deletion logic
        let fileManager = FileManager.default
        if let documentsUrl = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = documentsUrl.appendingPathComponent("yourFileName.txt") // Replace with your actual file name
            if fileManager.fileExists(atPath: fileURL.path) {
                do {
                    try fileManager.removeItem(at: fileURL)
                    print("File deleted successfully.")
                } catch {
                    print("Could not delete the file: \(error)")
                }
            } else {
                print("File does not exist.")
            }
        }
        print("Selected algorithms: \(selectedAlgorithms.map { algorithms[$0] })")
        DispatchQueue.global(qos: .userInitiated).async {
            var tempResults: [AlgorithmBenchmark] = []
            var baselineTiming: Double = 0.0
            
            for index in self.selectedAlgorithms.sorted() {
                let algorithm = self.algorithms[index]
                
                if let benchmarkFunction = self.msmBenchmarkMapping[algorithm] {
                    do {
                        print("Running MSM in algorithm: \(algorithm)...")
                        let benchData: BenchmarkResult =
                            try benchmarkFunction(
                                instanceSize,
                                numInstance,
                                documentsPath
                            )
                        
                        if algorithm == "Arkwork (Baseline)" {
                            baselineTiming = benchData.avgProcessingTime
                        }
                        
                        let algorithmBenchmark = AlgorithmBenchmark(algorithm: algorithm, avgMsmTime: benchData.avgProcessingTime, diffWithBaseline: (baselineTiming - benchData.avgProcessingTime) / baselineTiming * 100
                        )
                        tempResults.append(algorithmBenchmark)
                        print("Result of \(algorithmBenchmark.algorithm): \n \(algorithmBenchmark.avgMsmTime) ms (diff: \(algorithmBenchmark.diffWithBaseline) %)"
                        )
                    } catch {
                        print("Error running benchmark: \(error)")
                    }
                } else {
                    print("No benchmark function found for \(algorithm)")
                    tempResults.append(AlgorithmBenchmark(algorithm: algorithm, avgMsmTime: Double.nan, diffWithBaseline: Double.nan))
                }
            }

            DispatchQueue.main.async {
                self.benchmarkResults = tempResults
                self.isSubmitting = false
            }
        }
    }
}

