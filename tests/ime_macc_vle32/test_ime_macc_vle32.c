#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma riscv intrinsic "vector"

#define VLEN_BITS 16384
#define LAMBDA8_SELECTOR 4ULL
#define M 64
#define N 64
#define C_SLICE 512

static inline size_t
set_vtype_e32_lambda8(size_t lmul)
{
    uint64_t vtype;
    size_t vl;
    __asm__ volatile("csrr %[vtype], vtype" : [vtype] "=r" (vtype));
    vtype &= ~((7ULL << 0) | (7ULL << 3) | (7ULL << 60));
    vtype |= lmul == 8 ? 3ULL : 0ULL;
    vtype |= 2ULL << 3;
    vtype |= LAMBDA8_SELECTOR << 60;
    __asm__ volatile("vsetvl %[vl], zero, %[vtype]"
                     : [vl] "=r" (vl)
                     : [vtype] "r" (vtype));
    return vl;
}

static void
store_c_slices(int32_t c[8][C_SLICE])
{
    const size_t vl = C_SLICE;
    __asm__ volatile(
        "vse32.v v8,  (%[c0])\n"
        "vse32.v v9,  (%[c1])\n"
        "vse32.v v10, (%[c2])\n"
        "vse32.v v11, (%[c3])\n"
        "vse32.v v12, (%[c4])\n"
        "vse32.v v13, (%[c5])\n"
        "vse32.v v14, (%[c6])\n"
        "vse32.v v15, (%[c7])\n"
        :
        : [c0] "r" (&c[0][0]), [c1] "r" (&c[1][0]),
          [c2] "r" (&c[2][0]), [c3] "r" (&c[3][0]),
          [c4] "r" (&c[4][0]), [c5] "r" (&c[5][0]),
          [c6] "r" (&c[6][0]), [c7] "r" (&c[7][0]),
          [vl] "r" (vl)
        : "memory");
}

static void
run_vmmacc(int32_t *a, int32_t *b, int32_t c[8][C_SLICE])
{
    set_vtype_e32_lambda8(8);
    __asm__ volatile("vmv.v.i v8, 0"
                     :
                     :
                     : "memory", "v8", "v9", "v10", "v11", "v12",
                       "v13", "v14", "v15");
    set_vtype_e32_lambda8(1);
    __asm__ volatile(
        "vle32.v v0, (%[a])\n"
        "vle32.v v4, (%[b])\n"
        "vmmacc.vv v8, v0, v4\n"
        :
        : [a] "r" (a), [b] "r" (b)
        : "memory", "v0", "v4", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15");
    set_vtype_e32_lambda8(1);
    store_c_slices(c);
}

static void
run_vwmmacc(int16_t *a, int16_t *b, int32_t c[8][C_SLICE])
{
    set_vtype_e32_lambda8(8);
    __asm__ volatile("vmv.v.i v8, 0"
                     :
                     :
                     : "memory", "v8", "v9", "v10", "v11", "v12",
                       "v13", "v14", "v15");
    set_vtype_e32_lambda8(1);
    __asm__ volatile(
        "vle32.v v0, (%[a])\n"
        "vle32.v v4, (%[b])\n"
        "vwmmacc.vv v8, v0, v4\n"
        :
        : [a] "r" (a), [b] "r" (b)
        : "memory", "v0", "v4", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15");
    set_vtype_e32_lambda8(1);
    store_c_slices(c);
}

static void
run_vqwmmacc(int8_t *a, int8_t *b, int32_t c[8][C_SLICE])
{
    set_vtype_e32_lambda8(8);
    __asm__ volatile("vmv.v.i v8, 0"
                     :
                     :
                     : "memory", "v8", "v9", "v10", "v11", "v12",
                       "v13", "v14", "v15");
    set_vtype_e32_lambda8(1);
    __asm__ volatile(
        "vle32.v v0, (%[a])\n"
        "vle32.v v4, (%[b])\n"
        "vqwmmacc.vv v8, v0, v4\n"
        :
        : [a] "r" (a), [b] "r" (b)
        : "memory", "v0", "v4", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15");
    set_vtype_e32_lambda8(1);
    store_c_slices(c);
}

static int
check(const char *name, int32_t c[8][C_SLICE], int32_t expected[8][C_SLICE])
{
    for (size_t slice = 0; slice < 8; ++slice) {
        for (size_t i = 0; i < C_SLICE; ++i) {
            if (c[slice][i] != expected[slice][i]) {
                printf("%s: FAIL slice=%zu index=%zu got=%d expected=%d\n",
                       name, slice, i, c[slice][i], expected[slice][i]);
                return 0;
            }
        }
    }
    printf("%s: PASS\n", name);
    return 1;
}

static int
check_vmmacc(void)
{
    int32_t a[C_SLICE], b[C_SLICE];
    int32_t c[8][C_SLICE] = {};
    int32_t expected[8][C_SLICE] = {};
    for (size_t i = 0; i < M; ++i)
        for (size_t k = 0; k < 8; ++k)
            a[i * 8 + k] = 1 + (int32_t)((i + k) % 5);
    for (size_t j = 0; j < N; ++j)
        for (size_t k = 0; k < 8; ++k)
            b[j * 8 + k] = 1 + (int32_t)((2 * j + k) % 7);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            for (size_t k = 0; k < 8; ++k)
                expected[j / 8][i * 8 + j % 8] +=
                    a[i * 8 + k] * b[j * 8 + k];
    run_vmmacc(a, b, c);
    return check("vmmacc int32->int32", c, expected);
}

static int
check_vwmmacc(void)
{
    int16_t a[2 * C_SLICE], b[2 * C_SLICE];
    int32_t c[8][C_SLICE] = {};
    int32_t expected[8][C_SLICE] = {};
    for (size_t i = 0; i < M; ++i)
        for (size_t k = 0; k < 16; ++k)
            a[i * 16 + k] = (int16_t)(1 + (i + k) % 11);
    for (size_t j = 0; j < N; ++j)
        for (size_t k = 0; k < 16; ++k)
            b[j * 16 + k] = (int16_t)(1 + (2 * j + k) % 13);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            for (size_t k = 0; k < 16; ++k)
                expected[j / 8][i * 8 + j % 8] +=
                    a[i * 16 + k] * b[j * 16 + k];
    run_vwmmacc(a, b, c);
    return check("vwmmacc int16->int32", c, expected);
}

static int
check_vqwmmacc(void)
{
    int8_t a[4 * C_SLICE], b[4 * C_SLICE];
    int32_t c[8][C_SLICE] = {};
    int32_t expected[8][C_SLICE] = {};
    for (size_t i = 0; i < M; ++i)
        for (size_t k = 0; k < 32; ++k)
            a[i * 32 + k] = (int8_t)(1 + (i + k) % 17);
    for (size_t j = 0; j < N; ++j)
        for (size_t k = 0; k < 32; ++k)
            b[j * 32 + k] = (int8_t)(1 + (2 * j + k) % 19);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            for (size_t k = 0; k < 32; ++k)
                expected[j / 8][i * 8 + j % 8] +=
                    a[i * 32 + k] * b[j * 32 + k];
    run_vqwmmacc(a, b, c);
    return check("vqwmmacc int8->int32", c, expected);
}

int
main(void)
{
    int ok = check_vmmacc();
    ok &= check_vwmmacc();
    ok &= check_vqwmmacc();
    printf("IME MAC result: %s\n", ok ? "ALL PASSED" : "FAILED");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
