pub mod host;
pub mod metal_msm;
pub mod tests;
pub mod utils;

// Re-export main types for convenience
pub use metal_msm::{metal_variable_base_msm, MetalMSMConfig};

// Re-export test utilities
pub use metal_msm::test_utils;
