# Focused IME MAC test with VLE32 input loads

This test performs exactly one matrix multiplication for each integer widening
mode at VLEN=16384:

```text
vmmacc.vv   int32 x int32 -> int32
vwmmacc.vv  int16 x int16 -> int32
vqwmmacc.vv int8  x int8  -> int32
```

All A and B inputs are loaded only with `vle32.v`. The byte layout in the
loaded vector register is therefore the matrix layout consumed by IME:

- A is row-major: `A[i][k]`;
- B is column-major: `B[j][k]`, with each output column contiguous;
- C is initialized to zero;
- C uses eight adjacent vector registers `v8..v15`;
- each C register represents a 64x8 slice;
- each C slice is stored with one LMUL=1 `vse32.v`.

The VTYPE is written with whole-register `vsetvl`; lambda selector 4 in bits
62:60 represents lambda=8. The input/load and MAC state is SEW32, LMUL1.
The accumulator is cleared with SEW32, LMUL8 before returning to LMUL1 for the
input loads and MAC.

## Build and QEMU

```bash
export RV_TOOLCHAIN_WRAPPER="$HOME/Projects/rv-toolchain-wrapper"
export RV_CONTAINER_IMAGE="$HOME/Projects/IME/SDK/SIF/sdk-2026.06.29.sif"
export PATH="$RV_TOOLCHAIN_WRAPPER/bin:$PATH"
make
make qemu
```
