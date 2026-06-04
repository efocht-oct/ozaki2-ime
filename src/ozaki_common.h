#ifndef OZAKI_COMMON_H
#define OZAKI_COMMON_H

#include <stdint.h>

// Ozaki Scheme Parameters
#define NUM_MODULI 15

// Pre-selected coprime moduli near 256
extern const int32_t MODULI[NUM_MODULI];

// CRT Constants (Precomputed M_t and y_t where sum(C_t * M_t * y_t) % M = C)
// M = Product of all MODULI. Fits easily in __int128_t (~119 bits)
extern const __int128_t CRT_M; 
extern const __int128_t CRT_M_t_y_t[NUM_MODULI];

// Helper to compute (X * y) % M without 128-bit overflow
unsigned __int128 mult_mod(unsigned __int128 X, uint32_t y, unsigned __int128 M);

#endif // OZAKI_COMMON_H
