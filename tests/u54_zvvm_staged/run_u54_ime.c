#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M 64
#define N 64
#define K 32
#define NUM_MODULI 15
#define INPUT_LINE 8
#define A_PACKED (4 * M * INPUT_LINE)
#define B_PACKED (4 * N * INPUT_LINE)
#define C_TILE (M * N)
#define MAGIC 0x55563432U
#define LAMBDA8_SELECTOR 4ULL

static int8_t loaded_a[4 * M * INPUT_LINE];
static int8_t loaded_b[4 * N * INPUT_LINE];
static int32_t mac_c[8 * C_TILE];
static int skip_store;

struct file_header {
    uint32_t magic;
    uint32_t version;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t num_moduli;
};

static void
read_or_die(void *data, size_t size, size_t count, FILE *file)
{
    if (fread(data, size, count, file) != count) {
        perror("fread");
        exit(EXIT_FAILURE);
    }
}

static inline size_t
set_vtype_e32_lambda8(size_t lmul)
{
    uint64_t vtype;
    size_t vl;
    __asm__ volatile("csrr %[vtype], vtype" : [vtype] "=r" (vtype));
    vtype &= ~((7ULL << 0) | (7ULL << 3) | (7ULL << 60));
    vtype |= (lmul == 8 ? 3ULL : 0ULL) << 0;
    vtype |= 2ULL << 3;
    vtype |= LAMBDA8_SELECTOR << 60;
    __asm__ volatile("vsetvl %[vl], zero, %[vtype]"
                     : [vl] "=r" (vl)
                     : [vtype] "r" (vtype));
    return vl;
}

static inline void
run_ime_tile(int32_t *c, const int8_t *a, const int8_t *b)
{
    const size_t a_stride = INPUT_LINE;
    const size_t b_stride = N;
    const size_t c_stride = N;
    set_vtype_e32_lambda8(8);
    __asm__ volatile("vmv.v.i v8, 0"
                     :
                     :
                     : "memory", "v8", "v9", "v10", "v11", "v12",
                       "v13", "v14", "v15");
    set_vtype_e32_lambda8(1);
    __asm__ volatile(
        "vmtl.v v0, (%[a]), %[a_stride]\n"
        "vmttl.v v4, (%[b]), %[b_stride]\n"
        :
        : [a] "r" (a), [b] "r" (b),
          [a_stride] "r" (a_stride), [b_stride] "r" (b_stride)
        : "memory", "v0", "v4", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15");
    __asm__ volatile(
        "vse32.v v0, (%[a_dump])\n"
        "vse32.v v4, (%[b_dump])\n"
        :
        : [a_dump] "r" (loaded_a), [b_dump] "r" (loaded_b)
        : "memory");
    __asm__ volatile("vqwmmacc.vv v8, v0, v4"
                     :
                     :
                     : "memory", "v8", "v9", "v10", "v11", "v12",
                       "v13", "v14", "v15");
    set_vtype_e32_lambda8(1);
    __asm__ volatile(
        "vse32.v v8, (%[c0])\n"
        "vse32.v v9, (%[c1])\n"
        "vse32.v v10, (%[c2])\n"
        "vse32.v v11, (%[c3])\n"
        "vse32.v v12, (%[c4])\n"
        "vse32.v v13, (%[c5])\n"
        "vse32.v v14, (%[c6])\n"
        "vse32.v v15, (%[c7])\n"
        :
        : [c0] "r" (&mac_c[0 * C_TILE]),
          [c1] "r" (&mac_c[1 * C_TILE]),
          [c2] "r" (&mac_c[2 * C_TILE]),
          [c3] "r" (&mac_c[3 * C_TILE]),
          [c4] "r" (&mac_c[4 * C_TILE]),
          [c5] "r" (&mac_c[5 * C_TILE]),
          [c6] "r" (&mac_c[6 * C_TILE]),
          [c7] "r" (&mac_c[7 * C_TILE])
        : "memory");
    if (skip_store)
        return;
    set_vtype_e32_lambda8(8);
    __asm__ volatile("vmts.v v8, (%[c]), %[c_stride]"
                     :
                     : [c] "r" (c), [c_stride] "r" (c_stride)
                     : "memory", "v8", "v9", "v10", "v11", "v12",
                       "v13", "v14", "v15");
}

static int
check_loaded_tiles(const int8_t *a, const int8_t *b)
{
    for (size_t i = 0; i < sizeof(loaded_a); ++i) {
        if (loaded_a[i] != a[i]) {
            printf("vmtl load mismatch byte=%zu got=%d expected=%d\n",
                   i, loaded_a[i], a[i]);
            return 0;
        }
    }
    for (size_t j = 0; j < N; ++j) {
        for (size_t k = 0; k < K; ++k) {
            size_t source = (k / 4) * N * 4 + j * 4 + k % 4;
            size_t dest = j * K + k;
            if (loaded_b[dest] != b[source]) {
                printf("vmttl load mismatch byte=%zu got=%d expected=%d "
                       "source=%zu\n", dest, loaded_b[dest], b[source],
                       source);
                return 0;
            }
        }
    }
    printf("vmtl/vmttl loaded register images: PASS\n");
    return 1;
}

static int
check_mac_registers(const int32_t *expected)
{
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            size_t slice = j / 8;
            size_t slice_index = i * 8 + j % 8;
            if (mac_c[slice * C_TILE + slice_index] != expected[i * N + j]) {
                printf("vqwmmacc mismatch row=%zu col=%zu got=%d expected=%d\n",
                       i, j, mac_c[slice * C_TILE + slice_index],
                       expected[i * N + j]);
                return 0;
            }
        }
    }
    printf("vqwmmacc register image: PASS\\n");
    return 1;
}

int
main(int argc, char **argv)
{
    const char *input = argc > 1 ? argv[1] : "u54_case.bin";
    struct file_header header;
    static int8_t a[NUM_MODULI][A_PACKED];
    static int8_t b[NUM_MODULI][B_PACKED];
    static int32_t expected[NUM_MODULI][C_TILE];
    static int32_t actual[C_TILE];
    FILE *file = fopen(input, "rb");
    if (!file) {
        perror(input);
        return EXIT_FAILURE;
    }
    read_or_die(&header, sizeof(header), 1, file);
    if (header.magic != MAGIC || header.version != 1 ||
        header.m != M || header.n != N || header.k != K ||
        header.num_moduli != NUM_MODULI) {
        fprintf(stderr, "invalid test file header\n");
        fclose(file);
        return EXIT_FAILURE;
    }
    read_or_die(a, sizeof(a[0]), NUM_MODULI, file);
    read_or_die(b, sizeof(b[0]), NUM_MODULI, file);
    read_or_die(expected, sizeof(expected[0]), NUM_MODULI, file);
    fclose(file);

    printf("=== Staged U54 Zvvm IME test ===\n");
    printf("Loaded %s: M=%u N=%u K=%u moduli=%u\n",
           input, header.m, header.n, header.k, header.num_moduli);
    printf("Real Zvvm instructions: vmtl, vmttl, vqwmmacc, vmts\n");
    printf("Reference[0][0]=%d\n", expected[0][0]);
    printf("actual=%p expected=%p\n", (void *)actual,
           (void *)expected);

    skip_store = 1;
    for (unsigned t = 0; t < NUM_MODULI; ++t) {
        memset(actual, 0, sizeof(actual));
        run_ime_tile(actual, a[t], b[t]);
        if (t == 0 && !check_loaded_tiles(a[t], b[t]))
            return EXIT_FAILURE;
        if (t == 0 && !check_mac_registers(expected[t]))
            return EXIT_FAILURE;
        if (t == 0) {
            skip_store = 0;
            memset(actual, 0, sizeof(actual));
            run_ime_tile(actual, a[t], b[t]);
        }
        printf("after modulus %u: actual[0]=%d expected[0][0]=%d\\n",
               t, actual[0], expected[0][0]);
        for (unsigned i = 0; i < C_TILE; ++i) {
            if (actual[i] != expected[t][i]) {
                printf("FAIL: modulus=%u index=%u got=%d expected=%d\n",
                       t, i, actual[i], expected[t][i]);
                return EXIT_FAILURE;
            }
        }
    }

    printf("PASS: all %u modulus tiles match host reference\n", NUM_MODULI);
    return EXIT_SUCCESS;
}
