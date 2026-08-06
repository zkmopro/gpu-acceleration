// Segmented, load-balanced SMVP.
//
// The classic SMVP kernel assigns one thread per bucket, so a thread's work
// is its bucket's population — and the dispatch finishes when the fattest
// bucket does. Here threads own fixed-size slices of the (bucket-sorted)
// point stream instead, so per-thread work is uniform by construction:
//
//   smvp_accumulate: each thread scans stream positions [t*L, (t+1)*L) of
//     one subtask, accumulating runs of equal-row points. Because the stream
//     is sorted by row, any run that starts AND ends strictly inside the
//     slice covers its row completely -> written directly to row_sums
//     (exclusive, no atomics). Only the first and last run of a slice can be
//     partial -> spilled (at most 2 spills per thread).
//
//   smvp_merge: one thread per bucket pair (as before, but now doing O(1)
//     work per bucket plus one add per spill). A row's spills live in the
//     contiguous slice range [row_ptr[row]/L, (row_ptr[row+1]-1)/L], so no
//     searching is needed. Rows fitting inside a single slice were written
//     directly. Mirror rows are combined into signed buckets exactly like
//     the classic kernel.
#include "../curve/xyzz.metal"
#include <metal_stdlib>
using namespace metal;

#define SPILL_EMPTY 0xFFFFFFFFu

kernel void smvp_accumulate(
    device const uint* row_ptr [[buffer(0), access(read)]],
    device const uint* val_idx [[buffer(1), access(read)]],
    device const uint* chunks [[buffer(2), access(read)]],
    device const Affine* points [[buffer(3), access(read)]],
    device BigInt* row_x [[buffer(4), access(write)]],
    device BigInt* row_y [[buffer(5), access(write)]],
    device BigInt* row_zz [[buffer(6), access(write)]],
    device BigInt* row_zzz [[buffer(7), access(write)]],
    device BigInt* spill_x [[buffer(8), access(write)]],
    device BigInt* spill_y [[buffer(9), access(write)]],
    device BigInt* spill_zz [[buffer(10), access(write)]],
    device BigInt* spill_zzz [[buffer(11), access(write)]],
    device uint* spill_row [[buffer(12), access(write)]],
    constant uint4& params [[buffer(13), access(read)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint input_size = params[0];
    const uint num_columns = params[1];
    const uint seg_len = params[2];
    const uint subtask_offset = params[3];
    const uint num_segments = (input_size + seg_len - 1) / seg_len;

    const uint seg = gid.x;
    const uint local_subtask = gid.y; // chunk-local: row/spill buffers are reused per chunk
    const uint subtask = subtask_offset + local_subtask;
    if (seg >= num_segments) {
        return;
    }

    const uint rp_off = subtask * (num_columns + 1);
    const uint stream_off = subtask * input_size;
    const uint spill_base = (local_subtask * num_segments + seg) * 2;

    const uint seg_start = seg * seg_len;
    const uint seg_end = min(seg_start + seg_len, input_size);

    uint spills_used = 0;

    if (seg_start < input_size) {
        XYZZ acc = get_xyzz_zero();
        uint run_row = SPILL_EMPTY;
        uint run_start = seg_start;

        for (uint k = seg_start; k < seg_end; k++) {
            const uint idx = val_idx[stream_off + k];
            const uint row = chunks[stream_off + idx];

            if (row != run_row) {
                if (run_row != SPILL_EMPTY) {
                    // flush the finished run
                    const bool exclusive = (run_start == row_ptr[rp_off + run_row])
                        && (k == row_ptr[rp_off + run_row + 1]);
                    if (exclusive) {
                        const uint ri = local_subtask * num_columns + run_row;
                        row_x[ri] = acc.x;
                        row_y[ri] = acc.y;
                        row_zz[ri] = acc.zz;
                        row_zzz[ri] = acc.zzz;
                    } else {
                        const uint si = spill_base + spills_used;
                        spill_row[si] = run_row;
                        spill_x[si] = acc.x;
                        spill_y[si] = acc.y;
                        spill_zz[si] = acc.zz;
                        spill_zzz[si] = acc.zzz;
                        spills_used++;
                    }
                }
                acc = get_xyzz_zero();
                run_row = row;
                run_start = k;
            }
            acc = xyzz_madd_safe(acc, points[idx]);
        }

        // flush the trailing run
        if (run_row != SPILL_EMPTY) {
            const bool exclusive = (run_start == row_ptr[rp_off + run_row])
                && (seg_end == row_ptr[rp_off + run_row + 1]);
            if (exclusive) {
                const uint ri = local_subtask * num_columns + run_row;
                row_x[ri] = acc.x;
                row_y[ri] = acc.y;
                row_zz[ri] = acc.zz;
                row_zzz[ri] = acc.zzz;
            } else {
                const uint si = spill_base + spills_used;
                spill_row[si] = run_row;
                spill_x[si] = acc.x;
                spill_y[si] = acc.y;
                spill_zz[si] = acc.zz;
                spill_zzz[si] = acc.zzz;
                spills_used++;
            }
        }
    }

    // mark unused spill slots
    for (uint s = spills_used; s < 2; s++) {
        spill_row[spill_base + s] = SPILL_EMPTY;
    }
}

kernel void smvp_merge(
    device const uint* row_ptr [[buffer(0), access(read)]],
    device const BigInt* row_x [[buffer(1), access(read)]],
    device const BigInt* row_y [[buffer(2), access(read)]],
    device const BigInt* row_zz [[buffer(3), access(read)]],
    device const BigInt* row_zzz [[buffer(4), access(read)]],
    device const BigInt* spill_x [[buffer(5), access(read)]],
    device const BigInt* spill_y [[buffer(6), access(read)]],
    device const BigInt* spill_zz [[buffer(7), access(read)]],
    device const BigInt* spill_zzz [[buffer(8), access(read)]],
    device const uint* spill_row [[buffer(9), access(read)]],
    device BigInt* bucket_x [[buffer(10), access(write)]],
    device BigInt* bucket_y [[buffer(11), access(write)]],
    device BigInt* bucket_zz [[buffer(12), access(write)]],
    device BigInt* bucket_zzz [[buffer(13), access(write)]],
    constant uint4& params [[buffer(14), access(read)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint input_size = params[0];
    const uint num_columns = params[1];
    const uint seg_len = params[2];
    const uint subtask_offset = params[3];
    const uint num_segments = (input_size + seg_len - 1) / seg_len;

    const uint half_columns = num_columns / 2;
    const uint d = gid.x; // bucket-pair index within the subtask
    const uint local_subtask = gid.y;
    const uint subtask = subtask_offset + local_subtask;
    if (d >= half_columns) {
        return;
    }

    const uint rp_off = subtask * (num_columns + 1);
    const uint spill_off = local_subtask * num_segments * 2;

    XYZZ bucket = get_xyzz_zero();

    for (uint j = 0; j < 2; j++) {
        uint row = (j == 0) ? (d + half_columns) : (half_columns - d);
        if (j == 0 && d == 0) {
            row = 0;
        }

        const uint row_begin = row_ptr[rp_off + row];
        const uint row_end = row_ptr[rp_off + row + 1];

        XYZZ sum = get_xyzz_zero();
        if (row_begin < row_end) {
            const uint t0 = row_begin / seg_len;
            const uint t1 = (row_end - 1) / seg_len;
            if (t0 == t1) {
                // row fits inside one slice: written directly by its owner
                const uint ri = local_subtask * num_columns + row;
                sum.x = row_x[ri];
                sum.y = row_y[ri];
                sum.zz = row_zz[ri];
                sum.zzz = row_zzz[ri];
            } else {
                // row spans slices: all its partials are spills of slices t0..t1
                for (uint t = t0; t <= t1; t++) {
                    for (uint s = 0; s < 2; s++) {
                        const uint si = spill_off + t * 2 + s;
                        if (spill_row[si] == row) {
                            XYZZ part = {
                                .x = spill_x[si],
                                .y = spill_y[si],
                                .zz = spill_zz[si],
                                .zzz = spill_zzz[si]
                            };
                            sum = xyzz_add_safe(sum, part);
                        }
                    }
                }
            }
        }

        // signed-bucket mapping, as in the classic kernel
        uint bucket_idx = 0;
        if (half_columns > row) {
            bucket_idx = half_columns - row;
            sum = xyzz_neg(sum);
        } else {
            bucket_idx = row - half_columns;
        }

        if (bucket_idx > 0) {
            bucket = (j == 0) ? sum : xyzz_add_safe(bucket, sum);
        }
    }

    const uint bi = subtask * half_columns + d;
    bucket_x[bi] = bucket.x;
    bucket_y[bi] = bucket.y;
    bucket_zz[bi] = bucket.zz;
    bucket_zzz[bi] = bucket.zzz;
}
