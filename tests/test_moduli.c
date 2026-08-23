#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../src/ozaki_common.h"

static int check_basis(const int32_t *moduli, int count,
                       const __int128_t *constants, __int128_t product) {
    unsigned __int128 computed = 1;
    for (int i = 0; i < count; ++i) {
        if (moduli[i] < 2 || moduli[i] > 256) return 1;
        for (int j = 0; j < i; ++j)
            if (moduli[i] == moduli[j]) return 2;
        computed *= (unsigned)moduli[i];
    }
    if (computed != (unsigned __int128)product) return 3;
    for (int i = 0; i < count; ++i) {
        unsigned __int128 coefficient = (unsigned __int128)constants[i];
        if (coefficient % (unsigned)moduli[i] != 1) return 4;
        for (int j = 0; j < count; ++j)
            if (j != i && coefficient % (unsigned)moduli[j] != 0) return 5;
    }
    return 0;
}

int main(void) {
    if (FP64_NUM_MODULI != 15 || FP32_NUM_MODULI != 7) return 10;
    if (check_basis(MODULI_FP64, FP64_NUM_MODULI,
                    CRT_M_T_Y_T_FP64, CRT_M_FP64) != 0) return 11;
    if (check_basis(MODULI_FP32, FP32_NUM_MODULI,
                    CRT_M_T_Y_T_FP32, CRT_M_FP32) != 0) return 12;
    if (CRT_M_FP32 <= ((__int128_t)1 << 55)) return 13;
    puts("modulus basis and CRT constants: PASS (FP64=15, FP32=7)");
    return 0;
}
