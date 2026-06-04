# Ozaki-2 Scheme with RISC-V Integrated Matrix Extension (IME)

[![CI](https://github.com/efocht-oct/ozaki2-ime/actions/workflows/ci.yml/badge.svg)](https://github.com/efocht-oct/ozaki2-ime/actions/workflows/ci.yml)

This repository contains an implementation of the **Ozaki-2 scheme** for high-precision matrix multiplication (DGEMM and SGEMM) leveraging the **RISC-V Integrated Matrix Extension (IME / Zvvm family)**.

## What is the Ozaki-2 Scheme?

The Ozaki scheme is an error-free transformation technique for floating-point matrix multiplication. It allows exact computation of floating-point matrix products by:
1. **Scaling**: Extracting the maximum exponent of the input matrices and scaling the mantissas to integers.
2. **Modular Reduction**: Reducing the scaled integers modulo a set of pre-selected coprime moduli (e.g., near 256).
3. **Integer Matrix Multiplication**: Performing the matrix multiplication using high-throughput integer matrix-matrix multiplication (INT8 × INT8 → INT32) via hardware accelerators.
4. **CRT Reconstruction**: Reconstructing the exact integer result using the Chinese Remainder Theorem (CRT).
5. **Inverse Scaling**: Converting the exact integer back to floating-point and applying the inverse scale factor.

This approach avoids the accumulation of rounding errors inherent in standard floating-point MAC operations, providing bit-exact results for large matrix dimensions.

## RISC-V IME (Zvvm) Integration

The implementation utilizes the RISC-V Zvvm family of Integrated Matrix extensions, which accelerates matrix multiplication using the existing RISC-V Vector (V) register file, without introducing a separate architectural matrix register file.

Key IME features used:
- **Tile Geometry**: Configured via the `lambda` (λ) field in the `vtype` CSR.
- **Widening MAC**: `vqmmacc.vv` (quad-widening: INT8 × INT8 → INT32) and `v8wmmacc.vv` (8×-widening: INT4 × INT4 → INT32).
- **2D Tile Loads/Stores**: `vmtl.v` (order-preserving load), `vmttl.v` (transposing load for column-major data), and `vmts.v` (order-preserving store).

## Project Structure

```text
.
├── CLAUDE.md          # Context and guidelines for AI agents
├── Makefile           # Build system using rv-toolchain-wrapper
├── README.md          # This file
├── src/
│   ├── ozaki_dgemm.c  # Ozaki-2 DGEMM implementation (modern IME intrinsics)
│   └── ozaki_sgemm.c  # Ozaki-2 SGEMM implementation (modern IME intrinsics)
├── tests/
│   └── test_ozaki.c   # Test harness with CSR lambda configuration and VLEN testing
└── .github/
    └── workflows/
        └── ci.yml     # GitHub Actions CI using rv-toolchain-wrapper
```

## Prerequisites

- **Docker** or **Podman** (for the containerized toolchain)
- **Git**
- **Python 3** (for the `rv-toolchain-wrapper`)

## Build and Test

The project uses the `rv-toolchain-wrapper` to provide a consistent RISC-V cross-compilation and QEMU emulation environment.

```bash
# 1. Clone the rv-toolchain-wrapper (if not already present)
git clone https://github.com/efocht-oct/rv-toolchain-wrapper.git ~/Projects/IME/SDK/rv-toolchain-wrapper

# 2. Set up environment variables
export RV_CONTAINER_IMAGE=ghcr.io/efocht-oct/openchip/sdk:2026.05.28
export PATH=$HOME/Projects/IME/SDK/rv-toolchain-wrapper/bin:$PATH

# 3. Build the project
make clean
make all

# 4. Run the test suite (validates multiple VLEN and lambda configurations)
make test
```

## Configuration Parameters

The test suite explicitly configures the IME hardware via the `vtype` CSR before execution, as QEMU defaults to `lambda=1`. The primary target configuration validated is:
- **VLEN**: 16384
- **Lambda (λ)**: 8

This configuration yields an effective K-dimension of `K_eff = λ × W × LMUL = 8 × 4 × 1 = 32` for quad-widening (INT8) operations, perfectly matching the Ozaki-2 tile geometry.

## Expected DGEMM and SGEMM Performance

The DGEMM and SGEMM kernels in this repository are expected to track the sustained integer matrix-matrix performance of the IME unit, scaled by the number of modular products required by the Ozaki reconstruction. In the current implementation, both kernels use `NUM_MODULI = 15` coprime moduli near 256, and each modulus requires one complete INT8 × INT8 → INT32 matrix multiplication over the same `M × N × K` problem.

For a conventional GEMM, the useful floating-point work is counted as:

```text
FP work = 2 × M × N × K floating-point operations
```

where one multiply-add contributes two FLOPs. The Ozaki-IME path performs the same logical `M × N × K` multiply-adds once per modulus in integer arithmetic:

```text
Integer work = NUM_MODULI × M × N × K INT8 MACs
             = 15 × M × N × K INT8 MACs
```

Equivalently, a single IME tile for the configured geometry computes a `64 × 64` output tile over `K_eff = 32`, or:

```text
64 × 64 × 32 = 131,072 INT8 MACs per modulus
15 × 131,072 = 1,966,080 INT8 MACs per Ozaki tile result
```

If the sustained IME throughput is `P_int8_mac` INT8 MAC/s, the ideal large-matrix upper bound for both Ozaki DGEMM and Ozaki SGEMM is therefore:

```text
Expected FP MAC/s  ≈ P_int8_mac / 15
Expected FLOP/s    ≈ 2 × P_int8_mac / 15
```

If the IME performance number is reported as integer operations per second with one MAC counted as two integer operations, the equivalent estimate is:

```text
Expected GEMM FLOP/s ≈ P_int8_ops / 15
```

The estimate is identical for DGEMM and SGEMM in this code because both paths currently use the same 15-modulus CRT configuration and the same INT8 IME kernel. SGEMM can theoretically require fewer moduli than DGEMM because FP32 has a shorter significand, but this implementation keeps the same modular basis for both precisions to share the reconstruction path and provide a wide exact integer range.

These formulas are compute-bound estimates. Real measured performance will be lower when matrix sizes are small or when non-IME work is significant, including exponent scanning, scaling, modular reduction of `A` and `B`, memory allocation and packing, CRT reconstruction, inverse scaling, and the final `alpha`/`beta` update. These overheads are mostly proportional to `M × K`, `K × N`, or `M × N`; the IME multiplication term is proportional to `M × N × K`, so the estimate becomes more accurate for large, well-tiled matrices with sufficiently large `K`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
