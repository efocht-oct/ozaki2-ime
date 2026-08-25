# Ozaki2 IME Project

## Overview
This project implements the Ozaki-2 scheme for high-precision matrix multiplication (DGEMM/SGEMM) using the RISC-V Integrated Matrix Extension (IME / Zvvm family). The Ozaki scheme allows for error-free transformation of floating-point matrices into modular residues, performs the multiplication using exact FP8 digit matrix products, and reconstructs the exact floating-point result using the Chinese Remainder Theorem (CRT).

## Architecture
- **src/**: Contains the core Ozaki-2 DGEMM and SGEMM implementations using modern IME intrinsics.
- **tests/**: Contains test harnesses that validate the correctness of the implementations.
- **Makefile**: Build system leveraging the `rv-toolchain-wrapper` for containerized RISC-V cross-compilation and QEMU emulation.

## Key Concepts
- **Ozaki-2 Scheme**: Scales FP64/FP32 mantissas to integers, reduces them modulo a set of coprime moduli, performs exact two-digit FP8 matrix multiplication, and reconstructs the result via CRT.
- **IME (Zvvm)**: Uses 2D tile load/store (`vmtl.v`, `vmttl.v`, `vmts.v`) and widening matrix multiply-accumulate (`vqmmacc.vv`, `v8wmmacc.vv`) instructions.
- **Lambda (λ)**: Tile-layout parameter in the `vtype` CSR. Controls the geometry of the accumulator tile. Must be set via CSR write before executing IME instructions (QEMU defaults to λ=1).
- **VLEN**: Vector length. The implementation supports VLEN up to 16384.

## Build & Test
```bash
# Set up the toolchain wrapper
export RV_CONTAINER_IMAGE=ghcr.io/efocht-oct/openchip/sdk:2026.05.28
export PATH=~/Projects/IME/SDK/rv-toolchain-wrapper/bin:$PATH

# Build
make clean
make all

# Run tests
make test
```

## Code Standards
- Use modern RISC-V vector intrinsics (`<riscv_vector.h>`) instead of inline assembly where possible.
- Ensure `vtype` CSR is properly configured with the correct `lambda` value before any IME instruction.
- Matrix tiles are row-major. Use `vmtl.v` for row-major loads and `vmttl.v` for column-major to row-major transposing loads.
- Accumulator C is always signed. Use `altfmt_A` and `altfmt_B` in `vtype` to control input signedness.

## IME Intrinsics Reference
- `__riscv_vfqwmmacc_vv_f32m2_f8e4m3m1`: 4×-widening FP8 E4M3 → FP32 matrix MAC.
- `__riscv_v8wmmacc_vv_i32m4`: 8x-widening MAC (INT4 x INT4 -> INT32).
- `__riscv_vmtl_v_i32m2`: Order-preserving tile load.
- `__riscv_vmttl_v_i32m2`: Transposing tile load.
- `__riscv_vmts_v_i32m2`: Order-preserving tile store.

## Testing Requirements
- Tests must explicitly set the `lambda` CSR field before execution.
- Validate across multiple VLEN configurations (e.g., 1024, 4096, 16384).
- Validate across multiple lambda values (e.g., 1, 4, 8).
- The primary target configuration is **VLEN=16384, lambda=8**.
