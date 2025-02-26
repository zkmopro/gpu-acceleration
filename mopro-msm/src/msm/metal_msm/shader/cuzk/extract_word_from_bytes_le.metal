#pragma once

using namespace metal;
#include <metal_stdlib>
#include <metal_math>

uint32_t extract_word_from_bytes_le(
    const thread uint32_t* input,
    uint32_t word_idx,
    uint32_t chunk_size
) {
    uint32_t word = 0;
    const uint32_t start_byte_idx = 15 - ((word_idx * chunk_size + chunk_size) / 16);
    const uint32_t end_byte_idx = 15 - ((word_idx * chunk_size) / 16);
    
    const uint32_t start_byte_offset = (word_idx * chunk_size + chunk_size) % 16;
    const uint32_t end_byte_offset = (word_idx * chunk_size) % 16;
    
    uint32_t mask = 0;
    if (start_byte_offset > 0) {
        mask = (2 << (start_byte_offset - 1)) - 1;
    }
    if (start_byte_idx == end_byte_idx) {
        word = (input[start_byte_idx] & mask) >> end_byte_offset;
    } else {
        word = (input[start_byte_idx] & mask) << (16 - end_byte_offset);
        word += input[end_byte_idx] >> end_byte_offset;
    }
    
    return word;
}

// // Similar to bigint_funcs from bigint.metal
// BigInt extract_word_from_bytes_le(constant uint* bytes, uint word_idx, uint word_size) {
//     uint mask = (1 << word_size) - 1;
//     uint byte_idx = word_idx * word_size / 32;
//     uint bit_offset = (word_idx * word_size) % 32;
    
//     uint word = bytes[byte_idx];
//     if (bit_offset + word_size > 32) {
//         uint next_word = bytes[byte_idx + 1];
//         word = (next_word << (32 - bit_offset)) | (word >> bit_offset);
//     } else {
//         word = (word >> bit_offset) & mask;
//     }
//     return word;
// }