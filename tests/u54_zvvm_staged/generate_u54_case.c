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

static const int32_t moduli[NUM_MODULI] = {
    256, 255, 253, 251, 247, 239, 233, 229,
    227, 223, 217, 211, 199, 197, 193
};

struct file_header {
    uint32_t magic;
    uint32_t version;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t num_moduli;
};

static int8_t
symmetric_mod(uint64_t value, int32_t modulus)
{
    int64_t remainder = (int64_t)(value % (uint64_t)modulus);
    if (remainder > modulus / 2)
        remainder -= modulus;
    return (int8_t)remainder;
}

static uint64_t
input_value(unsigned index, unsigned salt)
{
    uint64_t x = 0x9e3779b97f4a7c15ULL * (index + 1 + salt);
    x ^= x >> 29;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 31;
    return x & ((1ULL << 54) - 1);
}

static uint64_t
pattern_a(unsigned pattern, unsigned i, unsigned k)
{
    if (pattern == 1)
        return i + 1;
    if (pattern == 2)
        return k + 1;
    if (pattern == 3)
        return 1 + ((i + 3 * k) % 7);
    return input_value(i * K + k, 17);
}

static uint64_t
pattern_b(unsigned pattern, unsigned j, unsigned k)
{
    if (pattern == 1)
        return k + 1;
    if (pattern == 2)
        return j + 1;
    if (pattern == 3)
        return 1 + ((5 * j + 2 * k) % 11);
    return input_value(k * N + j, 83);
}

static void
write_or_die(const void *data, size_t size, size_t count, FILE *file)
{
    if (fwrite(data, size, count, file) != count) {
        perror("fwrite");
        exit(EXIT_FAILURE);
    }
}

int
main(int argc, char **argv)
{
    const char *output = argc > 1 ? argv[1] : "u54_case.bin";
    unsigned pattern = argc > 2 ? (unsigned)atoi(argv[2]) : 0;
    uint64_t a[M * K];
    uint64_t b[K * N];
    int8_t a_logical[NUM_MODULI][M * K];
    int8_t b_logical[NUM_MODULI][N * K];
    int8_t a_packed[NUM_MODULI][A_PACKED];
    int8_t b_packed[NUM_MODULI][B_PACKED];
    int32_t reference[NUM_MODULI][C_TILE];
    struct file_header header = {0x55563432U, 1, M, N, K, NUM_MODULI};

    for (unsigned i = 0; i < M; ++i)
        for (unsigned k = 0; k < K; ++k)
            a[i * K + k] = pattern_a(pattern, i, k);
    for (unsigned k = 0; k < K; ++k)
        for (unsigned j = 0; j < N; ++j)
            b[k * N + j] = pattern_b(pattern, j, k);

    memset(a_packed, 0, sizeof(a_packed));
    memset(b_packed, 0, sizeof(b_packed));
    for (unsigned t = 0; t < NUM_MODULI; ++t) {
        for (unsigned i = 0; i < M; ++i) {
            for (unsigned k = 0; k < K; ++k) {
                int8_t value = symmetric_mod(a[i * K + k], moduli[t]);
                a_logical[t][i * K + k] = value;
                unsigned kk = k % INPUT_LINE;
                unsigned row = 4 * i + k / INPUT_LINE;
                a_packed[t][row * INPUT_LINE + kk] = value;
            }
        }
        for (unsigned j = 0; j < N; ++j) {
            for (unsigned k = 0; k < K; k += 4) {
                int8_t values[4];
                for (unsigned byte = 0; byte < 4; ++byte) {
                    unsigned logical_k = k + byte;
                    values[byte] = symmetric_mod(
                        b[logical_k * N + j], moduli[t]);
                    b_logical[t][j * K + logical_k] = values[byte];
                }
                // vmttl transposes an 8x64 matrix of 32-bit words:
                // source word (k/4, j) becomes destination word (j, k/4).
                unsigned word = (k / 4) * N + j;
                memcpy(&b_packed[t][word * sizeof(values)], values,
                       sizeof(values));
            }
        }
        for (unsigned i = 0; i < M; ++i) {
            for (unsigned j = 0; j < N; ++j) {
                int32_t sum = 0;
                for (unsigned k = 0; k < K; ++k)
                    sum += (int32_t)a_logical[t][i * K + k] *
                           (int32_t)b_logical[t][j * K + k];
                reference[t][i * N + j] = sum;
            }
        }
    }

    FILE *file = fopen(output, "wb");
    if (!file) {
        perror(output);
        return EXIT_FAILURE;
    }
    write_or_die(&header, sizeof(header), 1, file);
    write_or_die(a_packed, sizeof(a_packed[0]), NUM_MODULI, file);
    write_or_die(b_packed, sizeof(b_packed[0]), NUM_MODULI, file);
    write_or_die(reference, sizeof(reference[0]), NUM_MODULI, file);
    fclose(file);

    printf("Generated %s: M=%u N=%u K=%u moduli=%u pattern=%u\n",
           output, M, N, K, NUM_MODULI, pattern);
    printf("Packed A/B and int32 reference data are ready for RISC-V.\n");
    return 0;
}
