use mopro_ffi::{app, WtnsFn};

rust_witness::witness!(multiplier3);

app!();

fn zkey_witness_map(name: &str) -> Result<WtnsFn, MoproError> {
    match name {
        "multiplier3_final.zkey" => Ok(multiplier3_witness),
        _ => Err(MoproError::CircomError("Unknown circuit name".to_string())),
    }
}
