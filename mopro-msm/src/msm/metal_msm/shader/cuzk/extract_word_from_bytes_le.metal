#pragma once

#include <metal_math>
#include <metal_stdlib>
using namespace metal;

inline uint32_t extract_word_from_bytes_le(
    const thread uint32_t* input,
    uint32_t word_idx,
    uint32_t window_size)
{
    uint32_t word = 0;
    const uint32_t start_byte_idx = 15 - ((word_idx * window_size + window_size) / 16);
    const uint32_t end_byte_idx = 15 - ((word_idx * window_size) / 16);

    const uint32_t start_byte_offset = (word_idx * window_size + window_size) % 16;
    const uint32_t end_byte_offset = (word_idx * window_size) % 16;

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
