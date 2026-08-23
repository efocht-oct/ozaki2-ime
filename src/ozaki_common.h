#ifndef OZAKI_COMMON_H
#define OZAKI_COMMON_H

#include <stdint.h>

// DGEMM retains the original 15-modulus basis; SGEMM uses an independent
// 7-modulus basis.
#define FP64_NUM_MODULI 15
#define FP32_NUM_MODULI 7

extern const int32_t MODULI_FP64[FP64_NUM_MODULI];
extern const __int128_t CRT_M_FP64;
extern const __int128_t CRT_M_T_Y_T_FP64[FP64_NUM_MODULI];

extern const int32_t MODULI_FP32[FP32_NUM_MODULI];
extern const __int128_t CRT_M_FP32;
extern const __int128_t CRT_M_T_Y_T_FP32[FP32_NUM_MODULI];

// Helper to compute (X * y) % M without 128-bit overflow.
unsigned __int128 mult_mod(unsigned __int128 X, uint32_t y, unsigned __int128 M);

#endif // OZAKI_COMMON_H
