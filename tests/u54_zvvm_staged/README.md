# Staged U54 Zvvm test

This test splits the expensive U54 path into two stages:

1. `generate_u54_case` runs natively on the host and writes deterministic,
   packed IME inputs plus per-modulus int32 reference tiles to `u54_case.bin`.
2. `run_u54_ime.elf` runs as a RISC-V guest. It reads that file, executes only
   `vmtl.v`, `vmttl.v`, `vqwmmacc.vv`, and `vmts.v`, and compares every result
   against the host-generated reference.

The case is one 64x64 output tile with K=32 and 15 moduli. The packed input
layout matches VLEN=16384, SEW=32, lambda=8, LMUL=1.

The host generator accepts an optional pattern number:

```bash
./generate_u54_case u54_case_p1.bin 1
./generate_u54_case u54_case_p2.bin 2
./generate_u54_case u54_case_p3.bin 3
```

Patterns 1 and 2 make A/B row and column dependence intentionally asymmetric,
which helps distinguish B from B-transposed. Pattern 3 is a mixed ramp. The
reference data is computed from signed centered modular residues, matching the
default `altfmt_A=0` and `altfmt_B=0` integer IME interpretation.

QEMU does not implement these custom Zvvm opcodes and is expected to terminate
with `SIGILL`; the RISC-V runner is intended for gem5 or real IME hardware.

## Build

```bash
export RV_TOOLCHAIN_WRAPPER="$HOME/Projects/rv-toolchain-wrapper"
export RV_CONTAINER_IMAGE="$HOME/Projects/IME/SDK/SIF/sdk-2026.06.29.sif"
export PATH="$RV_TOOLCHAIN_WRAPPER/bin:$PATH"
make
```

## Native stage

```bash
make generate
```

## RISC-V/gem5 stage

The guest opens the binary file by path, so pass an absolute path when running
from gem5:

```bash
export GEM5_IME_LAMBDA_MODE=force
export GEM5_IME_LAMBDA_SEW8=32
export GEM5_IME_LAMBDA_SEW16=16
export GEM5_IME_LAMBDA_SEW32=8

/path/to/gem5_build/gem5.debug -d /tmp/staged-u54 \
  configs/ber/riscv-fs-dgemm.py \
  @configs/system_desc/ber200n_ct_1c.desc \
  --binary-path=$PWD/run_u54_ime.elf \
  --binary-arguments="$PWD/u54_case.bin"
```

The binary format is little-endian and contains a versioned header followed by
packed A, packed B, and int32 reference data for each modulus.

Tile leading dimensions are passed in elements, not bytes. In particular, the
int32 C store uses `N=64`, not `N * sizeof(int32_t)`.
