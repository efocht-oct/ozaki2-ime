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

The implementation utilizes the RISC-V Zvvm family of Integrated Matrix extensions, which accelerates matrix multiplication using the existing RISC-V Vector (V) register file, without introducing separate matrix registers.

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

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
