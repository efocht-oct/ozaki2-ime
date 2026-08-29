# vmts leading-dimension regression

This test isolates the Ozaki failure caused by passing an int32 leading
 dimension in bytes instead of elements to `vmts.v`.

The tile is 64x64 int32 values at VLEN=16384, SEW=32, LMUL=8, lambda=8.
The source tile is loaded with `vmtl.v`; the output is stored twice:

- `ld=64`: the correct leading dimension, measured in int32 elements.
- `ld=256`: the incorrect byte count (`64 * sizeof(int32_t)`) interpreted as
  an element leading dimension. This is expected to produce a sparse tile,
  not a compact 64x64 matrix.

The test passes only when both behaviors are observed. This makes the unit
explicitly test the erroneous caller convention rather than silently accepting
it as a valid compact store.

Build and run with QEMU:

```sh
make
make qemu
```

Run the same ELF with gem5 BER200N/double-mesh:

```sh
export GEM5_IME_LAMBDA_MODE=force
export GEM5_IME_LAMBDA_SEW8=32
export GEM5_IME_LAMBDA_SEW16=16
export GEM5_IME_LAMBDA_SEW32=8
unset GEM5_IME_LAMBDA_SEW64

LD_LIBRARY_PATH=/home/beegent/Projects/GEM5/gem5_vscratch/.pixi/envs/default/lib \
  /home/beegent/Projects/GEM5/gem5_vscratch/gem5_build/gem5.debug \
  -d /tmp/gem5-vmts-stride \
  /home/beegent/Projects/GEM5/gem5_vscratch/configs/ber/riscv-fs-dgemm.py \
  @/home/beegent/Projects/GEM5/gem5_vscratch/configs/system_desc/ber200n_ct_1c.desc \
  --binary-path=/home/beegent/Projects/ozaki2-ime/tests/vmts_stride_qemu_gem5/test_vmts_stride.elf
```
