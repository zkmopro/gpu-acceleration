# Guide for Extending Custom Functions Through Mopro

This guide will show you how to embed your own function into `mopro-ffi`, successfully generate either Swift or Kotlin bindings, and create a corresponding static library (e.g. xcframework if your target is iOS).

## Prerequisites

Before getting started, ensure you have:
1. Created your own function logic in Rust.
2. Forked the `mopro` repository.

## Import Your Crates in `mopro-ffi/src/lib.rs`

To generate Rust scaffolding, import your crates inside the `lib.rs` file in `mopro-ffi`. We recommend creating a replicate function in `lib.rs` that calls your functions from the imported crates. Then, create the same function inside the macro to get the relevant functions into scope and call FFI.

For example:

```rust
// mopro-ffi/src/lib.rs

use your_crates;

pub fn your_function(a: u32, b: u32) -> Result<bool, MoproError> {
    // Function logic
    your_crates::some_function()
}

#[macro_export]
macro_rules! app {
    () => {
        // Remember to import the struct here if you are using a custom one
        use mopro_ffi::{BenchmarkResult, GenerateProofResult, MoproError, ProofCalldata, G1, G2};
        use std::collections::HashMap;

        fn your_function(a: u32, b: u32) -> Result<bool, MoproError> {
            mopro_ffi::your_function(a, b)
        }

        uniffi::include_scaffolding!("mopro");
    };
}
```

### Using Feature Flags

If you are using a feature flag, handle the situation when the flag is not activated.

For example:

```rust
// mopro-ffi/src/lib.rs

use your_crates;

#[cfg(feature = "your-feature")]
pub fn your_function(a: u32, b: u32) -> Result<bool, MoproError> {
    // Function logic
    your_crates::some_function()
}

#[cfg(not(feature = "your-feature"))]
pub fn your_function(_: u32, _: u32) -> Result<bool, MoproError> {
    Err(MoproError::YourCustomError("something went wrong".to_string()))
}
```

## Extend the UDL File for Custom Functions

Modify `mopro-ffi/src/mopro.udl` to define your functions and variable types. Follow the [uniffi documentation](https://mozilla.github.io/uniffi-rs/0.28/udl/builtin_types.html) for more details.

For example:

```udl
// mopro-ffi/src/mopro.udl

namespace mopro {
  // ... other definitions

  [Throws=MoproError]
  boolean your_function(u32 a, u32 b);
};

dictionary YourStruct {
  u32 a;
  u32 b;
};
```

## Build Bindings for Swift and Kotlin

First, ensure the library name in `mopro-ffi/Cargo.toml` is "mopro_bindings". This guarantees the configuration is correctly set up for binding generation.

```toml
# mopro-ffi/Cargo.toml

[lib]
name = "mopro_bindings"  # Make sure the lib name is correct
```

Next, run the `mopro-ffi/build_bindings.sh` script to generate bindings for your custom functions. Remember to re-run it every time you modify the UDL file and push the resulting output to your repo.

## Access Custom Mopro Repo from Your App

Congratulations! Your custom functions have been successfully added to `mopro`. Now, let's integrate `mopro` into your app project. If you don't have an app project yet, we recommend forking one from [mopro-app](https://github.com/chancehudson/mopro-app).

First, follow the steps in [Mopro/Rust-setup](https://zkmopro.org/docs/getting-started/rust-setup) to create a valid setup.

Next, include your forked `mopro` dependencies and [uniffi](https://crates.io/crates/uniffi) in your Rust directory.

```toml
[package]
name = "your_app"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["lib", "cdylib", "staticlib"]
name = "mopro_bindings"  # This library name should not be changed

[[bin]]
name = "ios"

[features]
default = ["mopro-ffi/circom"]

[dependencies]
# ... other dependencies
mopro-ffi = { git = "https://github.com/zkmopro/mopro.git", branch = "feat/integrate-gpu-acceleration", features = ["gpu-acceleration"] }
uniffi = { version = "0.28", features = ["cli"] }

[build-dependencies]
# ... other dependencies
mopro-ffi = { git = "https://github.com/zkmopro/mopro.git", branch = "feat/integrate-gpu-acceleration", features = ["gpu-acceleration"] }
uniffi = { version = "0.28", features = ["build"] }
```

Finally, run the following command in the terminal to build the static library.

```sh
# CONFIGURATION is either debug or release
CONFIGURATION=release cargo run --bin ios
CONFIGURATION=debug cargo run --bin ios
```

Now, you can open `your-app/ios/your-app.xcodeproj` to check that your custom functions have been successfully bound to your app.
