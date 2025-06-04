use crate::msm::metal_msm::host::gpu::get_default_device;
// no runtime codegen of shader constants
use crate::msm::metal_msm::utils::metal_wrapper::{
    get_or_calc_constants, MSMConstants, MetalConfig,
};
use metal::*;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

// Include the single precompiled MSM metallib
include!(concat!(env!("OUT_DIR"), "/built_shaders.rs"));

/// Cache of compiled pipeline states
static PIPELINE_CACHE: Lazy<Mutex<HashMap<String, ComputePipelineState>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Shader types used in MSM pipeline
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShaderType {
    ConvertPointAndDecompose,
    Transpose,
    SMVP,
    BPRStage1,
    BPRStage2,
}

impl ShaderType {
    pub fn get_config(&self, num_limbs: usize, log_limb_size: u32) -> MetalConfig {
        match self {
            ShaderType::ConvertPointAndDecompose => MetalConfig {
                log_limb_size,
                num_limbs,
                shader_file: "cuzk/convert_point_coords_and_decompose_scalars.metal".to_string(),
                kernel_name: "convert_point_coords_and_decompose_scalars".to_string(),
            },
            ShaderType::Transpose => MetalConfig {
                log_limb_size,
                num_limbs,
                shader_file: "cuzk/transpose.metal".to_string(),
                kernel_name: "transpose".to_string(),
            },
            ShaderType::SMVP => MetalConfig {
                log_limb_size,
                num_limbs,
                shader_file: "cuzk/smvp.metal".to_string(),
                kernel_name: "smvp".to_string(),
            },
            ShaderType::BPRStage1 => MetalConfig {
                log_limb_size,
                num_limbs,
                shader_file: "cuzk/pbpr.metal".to_string(),
                kernel_name: "bpr_stage_1".to_string(),
            },
            ShaderType::BPRStage2 => MetalConfig {
                log_limb_size,
                num_limbs,
                shader_file: "cuzk/pbpr.metal".to_string(),
                kernel_name: "bpr_stage_2".to_string(),
            },
        }
    }
}

/// Configuration for shader manager
#[derive(Clone, Debug)]
pub struct ShaderManagerConfig {
    pub num_limbs: usize,
    pub log_limb_size: u32,
}

impl Default for ShaderManagerConfig {
    fn default() -> Self {
        Self {
            num_limbs: 16,
            log_limb_size: 16,
        }
    }
}

/// Pre-compiled shader information
#[derive(Clone)]
pub struct PrecompiledShader {
    pub pipeline_state: ComputePipelineState,
    pub config: MetalConfig,
    pub constants: MSMConstants,
}

/// Main shader manager that handles pre-compilation and caching
#[derive(Clone)]
pub struct ShaderManager {
    device: Device,
    config: ShaderManagerConfig,
    shaders: HashMap<ShaderType, PrecompiledShader>,
    constants: MSMConstants,
}

impl ShaderManager {
    /// Create a new shader manager with the given configuration
    pub fn new(config: ShaderManagerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let device = get_default_device();
        let constants = get_or_calc_constants(config.num_limbs, config.log_limb_size);
        let mut manager = Self {
            device,
            config: config.clone(),
            shaders: HashMap::new(),
            constants,
        };
        // Load precompiled library and build pipelines
        let library = manager.device.new_library_with_data(MSM_METALLIB)?;
        for shader_type in [
            ShaderType::ConvertPointAndDecompose,
            ShaderType::Transpose,
            ShaderType::SMVP,
            ShaderType::BPRStage1,
            ShaderType::BPRStage2,
        ] {
            let conf =
                shader_type.get_config(manager.config.num_limbs, manager.config.log_limb_size);
            let kernel = library.get_function(&conf.kernel_name, None)?;
            let ps = manager
                .device
                .new_compute_pipeline_state_with_function(&kernel)?;
            manager.shaders.insert(
                shader_type,
                PrecompiledShader {
                    pipeline_state: ps,
                    config: conf,
                    constants: manager.constants.clone(),
                },
            );
        }

        Ok(manager)
    }

    /// Create a shader manager with default configuration
    pub fn with_default_config() -> Result<Self, Box<dyn std::error::Error>> {
        Self::new(ShaderManagerConfig::default())
    }

    /// Get a precompiled shader by type
    pub fn get_shader(&self, shader_type: &ShaderType) -> Option<&PrecompiledShader> {
        self.shaders.get(shader_type)
    }

    /// Get the Metal device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the configuration
    pub fn config(&self) -> &ShaderManagerConfig {
        &self.config
    }

    /// Get the constants
    pub fn constants(&self) -> &MSMConstants {
        &self.constants
    }

    /// Clear the pipeline cache (useful for development/testing)
    pub fn clear_cache() {
        let mut cache = PIPELINE_CACHE.lock().unwrap();
        cache.clear();
    }
}

/// Builder pattern for creating shader manager with custom configuration
pub struct ShaderManagerBuilder {
    num_limbs: Option<usize>,
    log_limb_size: Option<u32>,
}

impl Default for ShaderManagerBuilder {
    fn default() -> Self {
        Self {
            num_limbs: None,
            log_limb_size: None,
        }
    }
}

impl ShaderManagerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_limbs(mut self, num_limbs: usize) -> Self {
        self.num_limbs = Some(num_limbs);
        self
    }

    pub fn log_limb_size(mut self, log_limb_size: u32) -> Self {
        self.log_limb_size = Some(log_limb_size);
        self
    }

    pub fn build(self) -> Result<ShaderManager, Box<dyn std::error::Error>> {
        let config = ShaderManagerConfig {
            num_limbs: self.num_limbs.unwrap_or(16),
            log_limb_size: self.log_limb_size.unwrap_or(16),
        };
        ShaderManager::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn test_shader_manager_creation() {
        let manager = ShaderManager::with_default_config().unwrap();
        assert_eq!(manager.config().num_limbs, 16);
        assert_eq!(manager.config().log_limb_size, 16);
    }

    #[test]
    #[serial_test::serial]
    fn test_shader_manager_builder() {
        let manager = ShaderManagerBuilder::new()
            .num_limbs(16)
            .log_limb_size(16)
            .build()
            .unwrap();

        assert_eq!(manager.config().num_limbs, 16);
        assert_eq!(manager.config().log_limb_size, 16);
    }

    #[test]
    #[serial_test::serial]
    fn test_get_precompiled_shaders() {
        let manager = ShaderManager::with_default_config().unwrap();

        let convert_shader = manager.get_shader(&ShaderType::ConvertPointAndDecompose);
        assert!(convert_shader.is_some());

        let transpose_shader = manager.get_shader(&ShaderType::Transpose);
        assert!(transpose_shader.is_some());

        let smvp_shader = manager.get_shader(&ShaderType::SMVP);
        assert!(smvp_shader.is_some());
    }
}
