#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VLEN_BITS 16384
#define ROWS 64
#define COLS 64
#define ELEMENT_STRIDE COLS
#define BYTE_STRIDE (COLS * sizeof(int32_t))
#define STORAGE_SENTINEL INT32_C(0x5a5a5a5a)
#define LAMBDA8_SELECTOR 4ULL

static size_t
set_vtype_e32_m8_lambda8(void)
{
    uint64_t vtype;
    size_t vl;

    __asm__ volatile("csrr %[vtype], vtype"
                     : [vtype] "=r" (vtype));
    vtype &= ~((7ULL << 0) | (7ULL << 3) | (7ULL << 60));
    vtype |= 3ULL;                 /* LMUL=8 */
    vtype |= 2ULL << 3;            /* SEW=32 */
    vtype |= LAMBDA8_SELECTOR << 60;
    __asm__ volatile("vsetvl %[vl], zero, %[vtype]"
                     : [vl] "=r" (vl)
                     : [vtype] "r" (vtype));
    return vl;
}

static void
load_tile(int32_t *src)
{
    const size_t vl = set_vtype_e32_m8_lambda8();
    __asm__ volatile("vmtl.v v8, (%[src]), %[stride]"
                     :
                     : [src] "r" (src),
                       [stride] "r" ((size_t)ELEMENT_STRIDE)
                     : "memory", "v8", "v9", "v10", "v11",
                       "v12", "v13", "v14", "v15");
    (void)vl;
}

static void
store_tile(int32_t *dst, size_t leading_dimension)
{
    set_vtype_e32_m8_lambda8();
    __asm__ volatile("vmts.v v8, (%[dst]), %[stride]"
                     :
                     : [dst] "r" (dst), [stride] "r" (leading_dimension)
                     : "memory", "v8", "v9", "v10", "v11",
                       "v12", "v13", "v14", "v15");
}

static int
check_tile(const int32_t *src, const int32_t *dst, size_t stride)
{
    for (size_t r = 0; r < ROWS; ++r) {
        for (size_t c = 0; c < COLS; ++c) {
            const int32_t expected = src[r * ELEMENT_STRIDE + c];
            if (dst[r * stride + c] != expected) {
                printf("mismatch row=%zu col=%zu got=%d expected=%d\n",
                       r, c, dst[r * stride + c], expected);
                return 0;
            }
        }
    }
    return 1;
}

static int
check_compact_mismatch(const int32_t *src, const int32_t *dst)
{
    size_t mismatches = 0;
    for (size_t r = 0; r < ROWS; ++r) {
        for (size_t c = 0; c < COLS; ++c) {
            if (dst[r * ELEMENT_STRIDE + c] !=
                src[r * ELEMENT_STRIDE + c])
                ++mismatches;
        }
    }
    printf("byte-stride compact mismatches=%zu, row1col0=%d, "
           "sparse-row1col0=%d\n", mismatches, dst[COLS], dst[BYTE_STRIDE]);
    return mismatches != 0;
}

int
main(void)
{
    const size_t good_count = ROWS * ELEMENT_STRIDE;
    const size_t bad_count = (ROWS - 1) * BYTE_STRIDE + COLS;
    int32_t *src = calloc(good_count, sizeof(*src));
    int32_t *dst_good = calloc(good_count, sizeof(*dst_good));
    int32_t *dst_bad = malloc(bad_count * sizeof(*dst_bad));
    if (!src || !dst_good || !dst_bad)
        return EXIT_FAILURE;

    for (size_t r = 0; r < ROWS; ++r)
        for (size_t c = 0; c < COLS; ++c)
            src[r * ELEMENT_STRIDE + c] = 100000 + r * COLS + c;
    for (size_t i = 0; i < bad_count; ++i)
        dst_bad[i] = STORAGE_SENTINEL;

    load_tile(src);

    store_tile(dst_good, ELEMENT_STRIDE);
    const int good_ok = check_tile(src, dst_good, ELEMENT_STRIDE);
    printf("vmts leading dimension in words (%zu): %s\n",
           (size_t)ELEMENT_STRIDE, good_ok ? "PASS" : "FAIL");

    store_tile(dst_bad, BYTE_STRIDE);
    const int bad_compact_mismatch = check_compact_mismatch(src, dst_bad);
    const int bad_ok = bad_compact_mismatch;
    printf("vmts leading dimension in bytes (%zu): %s\n",
           (size_t)BYTE_STRIDE,
           bad_ok ? "EXPECTED-MISMATCH" : "UNEXPECTED-COMPACT");
    printf("vmts stride result: %s\n", good_ok && bad_ok ?
           "ALL PASSED" : "FAILED");

    free(src);
    free(dst_good);
    free(dst_bad);
    return good_ok && bad_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
