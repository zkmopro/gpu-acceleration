import XCTest
@testable import gpu_acceleration

class MSMBenchmarkViewUITests: XCTestCase {
    
    var app: XCUIApplication!
    
    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
    }
    
    func testMSMBenchmarkViewUI() throws {
        // Navigate to MSMBenchmarkView
        app.buttons["Generate Benchmarks"].tap()
        
        // Check if the navigation title is correct
        XCTAssertTrue(app.navigationBars["MSM Benchmark"].exists)
        
        // Check if both algorithms are listed
        XCTAssertTrue(app.staticTexts["1. Arkwork (Baseline)"].exists)
        XCTAssertTrue(app.staticTexts["2. Metal (GPU)"].exists)
        
        // Check if the "Generate Benchmarks" button exists
        XCTAssertTrue(app.buttons["Generate Benchmarks"].exists)
        
        // Check if the results section exists
        XCTAssertTrue(app.staticTexts["BENCHMARK RESULTS"].exists)
        
        // Tap on the Metal (GPU) algorithm to select it
        app.staticTexts["2. Metal (GPU)"].tap()
        
        // Verify that both algorithms are now selected (checkmarks exist)
        XCTAssertEqual(app.images.matching(identifier: "checkmark").count, 2)
        
        // Tap the "Generate Benchmarks" button
        app.buttons["Generate Benchmarks"].tap()
    }
}
