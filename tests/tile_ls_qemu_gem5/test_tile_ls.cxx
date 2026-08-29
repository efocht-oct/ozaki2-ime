#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma riscv intrinsic "vector"

#define VLEN_BITS 16384
#define LAMBDA8_SELECTOR 4ULL
#define LAMBDA16_SELECTOR 5ULL

static inline size_t
set_vtype(size_t sew_log2, size_t lmul_encoding, uint64_t lambda_selector)
{
    uint64_t vtype;
    size_t vl;
    __asm__ volatile("csrr %[vtype], vtype" : [vtype] "=r" (vtype));
    vtype &= ~((7ULL << 0) | (7ULL << 3) | (7ULL << 60));
    vtype |= lmul_encoding;
    vtype |= sew_log2 << 3;
    vtype |= lambda_selector << 60;
    __asm__ volatile("vsetvl %[vl], zero, %[vtype]"
                     : [vl] "=r" (vl)
                     : [vtype] "r" (vtype));
    return vl;
}

static int
check_i64_single(void)
{
    const size_t sew = 64;
    const size_t lambda = 16;
    const size_t rows = VLEN_BITS / (sew * lambda);
    const size_t cols = lambda;
    const size_t stride = cols + 2;
    int64_t *src = (int64_t *)calloc(rows * stride, sizeof(*src));
    int64_t *dst = (int64_t *)calloc(rows * stride, sizeof(*dst));
    if (!src || !dst)
        return 0;
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            src[r * stride + c] = (int64_t)(1000 + r * cols + c);

    size_t vl = set_vtype(3, 0, LAMBDA16_SELECTOR);
    vint64m1_t v = __riscv_vmtl_v_i64m1(src, (ptrdiff_t)stride, vl);
    set_vtype(3, 0, LAMBDA16_SELECTOR);
    __riscv_vmts_v_i64m1(dst, (ptrdiff_t)stride, v, vl);

    int ok = 1;
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            ok &= dst[r * stride + c] == src[r * stride + c];
    printf("single-register SEW64 LMUL1 lambda16: %s\n",
           ok ? "PASS" : "FAIL");
    free(src);
    free(dst);
    return ok;
}

static int
check_i32_multi(void)
{
    const size_t sew = 32;
    const size_t lambda = 8;
    const size_t lmul = 8;
    const size_t rows = VLEN_BITS / (sew * lambda);
    const size_t cols = lambda * lmul;
    const size_t stride = cols + 2;
    int32_t *src = (int32_t *)calloc(rows * stride, sizeof(*src));
    int32_t *dst = (int32_t *)calloc(rows * stride, sizeof(*dst));
    if (!src || !dst)
        return 0;
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            src[r * stride + c] = (int32_t)(2000 + r * cols + c);

    size_t vl = set_vtype(2, 3, LAMBDA8_SELECTOR);
    vint32m8_t v = __riscv_vmtl_v_i32m8(src, (ptrdiff_t)stride, vl);
    set_vtype(2, 3, LAMBDA8_SELECTOR);
    __riscv_vmts_v_i32m8(dst, (ptrdiff_t)stride, v, vl);

    int ok = 1;
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            ok &= dst[r * stride + c] == src[r * stride + c];
    printf("multi-register SEW32 LMUL8 lambda8: %s\n",
           ok ? "PASS" : "FAIL");
    free(src);
    free(dst);
    return ok;
}

static int
check_i32_lmul_switch(bool load_multi)
{
    const size_t rows = 64;
    const size_t cols = 64;
    const size_t common_cols = 8;
    const size_t stride = cols + 2;
    int32_t *src = (int32_t *)calloc(rows * stride, sizeof(*src));
    int32_t *dst = (int32_t *)calloc(rows * stride, sizeof(*dst));
    if (!src || !dst)
        return 0;
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            src[r * stride + c] = (int32_t)(3000 + r * cols + c);

    if (load_multi) {
        set_vtype(2, 3, LAMBDA8_SELECTOR);
        __asm__ volatile("vmtl.v v8, (%[src]), %[stride]"
                         :
                         : [src] "r" (src), [stride] "r" (stride)
                         : "memory", "v8", "v9", "v10", "v11",
                           "v12", "v13", "v14", "v15");
        set_vtype(2, 0, LAMBDA8_SELECTOR);
        __asm__ volatile("vmts.v v8, (%[dst]), %[stride]"
                         :
                         : [dst] "r" (dst), [stride] "r" (stride)
                         : "memory", "v8");
    } else {
        set_vtype(2, 3, LAMBDA8_SELECTOR);
        __asm__ volatile("vmv.v.i v8, 0"
                         :
                         :
                         : "memory", "v8", "v9", "v10", "v11",
                           "v12", "v13", "v14", "v15");
        set_vtype(2, 0, LAMBDA8_SELECTOR);
        __asm__ volatile("vmtl.v v8, (%[src]), %[stride]"
                         :
                         : [src] "r" (src), [stride] "r" (stride)
                         : "memory", "v8");
        set_vtype(2, 3, LAMBDA8_SELECTOR);
        __asm__ volatile("vmts.v v8, (%[dst]), %[stride]"
                         :
                         : [dst] "r" (dst), [stride] "r" (stride)
                         : "memory", "v8", "v9", "v10", "v11",
                           "v12", "v13", "v14", "v15");
    }

    int ok = 1;
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < common_cols; ++c)
            ok &= dst[r * stride + c] == src[r * stride + c];
    printf("load LMUL%u -> store LMUL%u (SEW32 lambda8): %s\n",
           load_multi ? 8 : 1, load_multi ? 1 : 8,
           ok ? "PASS" : "FAIL");
    free(src);
    free(dst);
    return ok;
}

int
main(void)
{
    int ok = check_i64_single();
    ok &= check_i32_multi();
    ok &= check_i32_lmul_switch(false);
    ok &= check_i32_lmul_switch(true);
    printf("tile load/store result: %s\n", ok ? "ALL PASSED" : "FAILED");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
