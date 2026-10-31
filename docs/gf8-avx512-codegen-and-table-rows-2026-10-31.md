# GF8 AVX-512BW codegen and split-table row kernels

This note records the design, emulation verification, and native `RustAvx2`
performance screen for the `GF(2^8)` AVX-512BW whole-codeword encode codegen,
the `6+2` generated encode shape, and the bounded `table_rows` multi-input
dot-product kernels (`src/galois_8/x86/table_rows.rs`).

## Scope and generated-code observations

- `build.rs` and `src/galois_8/x86/codegen.rs` add `avx512f` + `avx512bw`
  generated encode kernels (`encode_4x2_avx512` through `encode_12x4_avx512`)
  using 64-byte `_mm512_shuffle_epi8` nibble lookups and `_mm512_ternarylogic_epi64`
  (`0x96`) three-way XOR fusion when `active_backend_id() == BackendId::RustAvx512`
  (or in AVX-512-only builds without `simd-avx2`). The `6+2` geometry is added
  to the AVX2, AVX-512BW, and AArch64 NEON generated shape tables.
- `src/galois_8/x86/table_rows.rs` adds `SelectedTableRowKernel` (`dot_table_avx2`
  and `dot_table_avx512`) for `RustAvx2` and `RustAvx512` across `inputs in 4..=16`,
  `outputs in 1..=4`, and `shard_len >= 64`. Each output row keeps four vector
  accumulators (`128 B` per step on AVX2, `256 B` per step on AVX-512BW) in
  registers across all survivor inputs, eliminating `d - 1` intermediate
  read-modify-write passes over the output buffer during reconstruction and
  non-codegen encoding.
- `TableTerm` pre-caches the 16-byte `low_tbl` and `high_tbl` pointers once per
  call before the 128-byte/256-byte chunk loop, and `SelectedTableRowKernel::select`
  bounds `inputs` to `4..=16`. Wider configurations such as `32+16` remain on
  the outer coefficient-streaming loop to avoid L1 data cache and hardware
  prefetcher thrashing across 32 simultaneous input streams.
- `src/galois_8/x86/gfni_rows.rs` retains `!(2..=4).contains(&outputs)` so
  single-output GFNI operations continue to use the `code_single_slice`
  coefficient-streaming path verified by
  `single_output_gfni_entries_keep_coefficient_streaming`.
- On the observed Linux x86-64 release build (`objdump -d -M intel`),
  `encode_12x4_avx2` spills lookup tables to `[rsp+0x100..0x1e0]` inside the
  32-byte hot loop (`217` stack-relative references), whereas
  `encode_12x4_avx512` keeps all 24 coefficient tables in `zmm1..zmm25` with
  zero stack table reloads in the 64-byte loop and fuses low/high nibble XOR
  reductions with `vpternlogq $0x96`. Generated code is compiler- and
  target-dependent.

## Functional and emulation evidence

- On the native Linux x86-64 host (`AMD Ryzen Threadripper PRO 3945WX`,
  `SSSE3=true, AVX2=true, AVX512F=false, AVX512BW=false, GFNI=false`), the full
  `--all-features` library and integration suites, `no_std` library tests,
  `--no-default-features --features "std simd-avx512"` build and integration
  tests, Clippy (`--all-targets --all-features -- -D warnings`), and `xtask`
  repository audits (`audit-json`, `audit-no-python`, `audit-package`,
  `audit-workspace`) passed.
- Offline `qemu-x86_64` (version 11.1.0) user-mode emulation passed the
  `galois_8` library and dispatch regression suites under `-cpu max`
  (`SSSE3=true, AVX2=true`) and `-cpu Westmere` (`SSSE3=true, AVX2=false`).
  However, QEMU's x86_64 TCG backend does not implement `avx512f`, `avx512bw`,
  or `gfni` (`warning: TCG doesn't support requested feature: CPUID[eax=07h,ecx=00h].EBX.avx512f [bit 16]`)
  and strips those bits from `CPUID`, causing runtime AVX-512 feature checks to
  skip under QEMU.
- To execute the `AVX-512BW` and `GFNI` paths end-to-end on this host, the
  release test binaries were run under **Intel® Software Development Emulator
  (`sde64` 10.13.1)** with `-skx` (`AVX512F=true, AVX512BW=true, GFNI=false`)
  and `-spr` (`AVX512F=true, AVX512BW=true, GFNI=true`):
  - `galois_8::x86::codegen::tests::generated_avx512_bytes_match_scalar_when_supported`
    passed across all seven shapes (`4+2`, `6+2`, `6+3`, `8+3`, `8+4`, `10+4`,
    `12+4`) and seven boundary lengths (`1, 31, 32, 63, 64, 65, 65536`),
    executing `373,464` 512-bit `VPSHUFB` and `164,160` 512-bit `VPTERNLOGQ`
    instructions under `sde64 -skx -mix`.
  - `galois_8::x86::table_rows::tests::table_rows_match_scalar_for_avx2_and_avx512`
    passed across `inputs in [4, 6, 8, 12, 16]`, `outputs in [1, 2, 4]`, and
    13 shard lengths (`0..65536`), executing `9,858` 512-bit `VPSHUFB` and
    `4,929` 512-bit `VPTERNLOGQ` instructions.
  - `galois_8::tests::test_x86_cross_backend_conformance_matrix`,
    `galois_8::tests::test_active_backend_metadata`,
    `tests/encoding_dispatch_regressions.rs` (including
    `native::single_output_gfni_entries_keep_coefficient_streaming` under
    `-spr`), and `tests/parallel_codegen_regressions.rs` passed under both
    `-skx` and `-spr`.
  - `tests/benchmark_smoke.rs` (`benchmark_smoke_matrix_runs_and_exports_results`)
    passed with `RSE_STRICT_BACKEND_OVERRIDE=1` for `scalar-rust`, `rust-avx2`,
    `rust-avx512` (`-skx`), and `rust-gfni-avx512` (`-spr`).

The table below summarizes the guest dynamic instruction counts captured by
`sde64 -skx -mix` across the parallel codegen regression suite (`simd-avx2` vs
`simd-avx512` builds) and a single-iteration `benchmark_smoke` run. Intel SDE
is functional and instruction-mix evidence only, not native AVX-512 hardware
wall-clock timing.

| Workload under `sde64 -skx -mix` | Path / Build | Total dynamic instructions | `VPSHUFB` instructions | `VPTERNLOGQ` instructions |
| --- | --- | ---: | ---: | ---: |
| `parallel_codegen_regressions` (`7` shapes) | `simd-avx2` | `483,936,840` | `29,688,536` | `0` |
| `parallel_codegen_regressions` (`7` shapes) | `simd-avx512` | `389,732,916` (−19.47%) | `14,843,904` (−50.00%) | `1,462,398` (`VPXORQ`) / fused |
| `benchmark_smoke` (`1` iter, strict override) | `scalar-rust` | `2,440,237,807` | `0` | `0` |
| `benchmark_smoke` (`1` iter, strict override) | `rust-avx2` | `1,219,690,987` | `18,630,656` | `0` |
| `benchmark_smoke` (`1` iter, strict override) | `rust-avx512` | `1,165,601,761` (−4.43% vs `avx2`) | `9,315,328` (−50.00%) | `4,653,056` |

## Native Linux x86-64 performance screen (`RustAvx2`)

On 2026-10-31 the baseline at `a0f74a0` was compared against the candidate on
the `AMD Ryzen Threadripper PRO 3945WX` host (`RustAvx2`, `std` + `simd-accel`).
The [recorded measurements](../benchmarks/gf8-avx512-codegen-and-table-rows/2026-10-31-linux-x86/measurements.json)
retain the `benches/recovery_paths` Criterion medians/samples, the 10-iteration
extended `benchmark_smoke` throughput/latency figures, and the `sde64`
instruction-mix summaries. Negative latency deltas and positive throughput
deltas indicate faster execution.

| Benchmark suite and case | Shard size | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: | ---: |
| `recovery_paths`: `reconstruct_data_slots/4x2/missing1+0` | 16 KiB | 1.68 µs | 1.58 µs | −5.86% latency |
| `recovery_paths`: `reconstruct_data_slots/4x2/missing2+0` | 16 KiB | 3.17 µs | 3.00 µs | −5.47% latency |
| `recovery_paths`: `reconstruct_data_vec/4x2/missing1+0` | 16 KiB | 1.83 µs | 1.76 µs | −3.98% latency |
| `recovery_paths`: `reconstruct_data_vec/4x2/missing2+0` | 16 KiB | 3.52 µs | 3.39 µs | −3.52% latency |
| `recovery_paths`: `reconstruct_data_opt/10x4/missing1+0` | 32 KiB | 8.92 µs | 8.53 µs | −4.35% latency |
| `recovery_paths`: `reconstruct_data_opt/10x4/missing2+0` | 32 KiB | 17.36 µs | 17.08 µs | −1.63% latency |
| `recovery_paths`: `reconstruct_data_opt/10x4/missing1+0` | 1 MiB | 216.32 µs | 192.33 µs | −11.09% latency |
| `recovery_paths`: `reconstruct_some_parity/32x16/missing1+1` | 64 KiB | 53.69 µs | 54.13 µs | +0.81% latency |
| `benchmark_smoke`: `reconstruct/4x2` | 64 KiB | 625.87 MB/s | 654.38 MB/s | +4.56% throughput |
| `benchmark_smoke`: `reconstruct_data/4x2` | 64 KiB | 612.46 MB/s | 647.26 MB/s | +5.68% throughput |
| `benchmark_smoke`: `verify/4x2` | 64 KiB | 600.80 MB/s | 619.91 MB/s | +3.18% throughput |
| `benchmark_smoke`: `reconstruct/32x16` | 1 MiB | 456.32 MB/s | 457.35 MB/s | +0.23% throughput |
| `benchmark_smoke`: `reconstruct_data/32x16` | 1 MiB | 456.93 MB/s | 458.23 MB/s | +0.29% throughput |

These measurements are a local workstation screen on native `RustAvx2` hardware
combined with Intel SDE `AVX-512BW` / `GFNI` instruction-level emulation, not a
formal release qualification on physical `AVX-512` silicon.
