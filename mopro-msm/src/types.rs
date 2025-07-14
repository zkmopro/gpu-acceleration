#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

pub mod raw {
    // 1. pull in the bindgen output that build.rs dropped in OUT_DIR
    include!(concat!(env!("OUT_DIR"), "/common.rs"));

    // 2. add Pod/Zeroable so we can cast to &[u8] safely
    use bytemuck::{Pod, Zeroable};

    unsafe impl Zeroable for Uniforms {}
    unsafe impl Pod for Uniforms {}
    unsafe impl Zeroable for Params {}
    unsafe impl Pod for Params {}
}
