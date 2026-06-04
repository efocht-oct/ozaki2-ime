#include "ozaki_common.h"

// Pre-selected coprime moduli near 256
const int32_t MODULI[NUM_MODULI] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193
};

// CRT Constants (Precomputed M_t and y_t where sum(C_t * M_t * y_t) % M = C)
// M = Product of all MODULI. Fits easily in __int128_t (~119 bits)
const __int128_t CRT_M = (((__int128_t)12413391266671370ULL << 64) | 17550328509527814400ULL);

const __int128_t CRT_M_t_y_t[NUM_MODULI] = {
    (((__int128_t)824326763802395ULL << 64) | 13415242489033580545ULL), // m=256
    (((__int128_t)8713713869545785ULL << 64) | 16298351871605898496ULL), // m=255
    (((__int128_t)9322309646907353ULL << 64) | 12523880316786161152ULL), // m=253
    (((__int128_t)9248223772380662ULL << 64) | 15133148467512226048ULL), // m=251
    (((__int128_t)8945682775172081ULL << 64) | 1669178525751606784ULL), // m=247
    (((__int128_t)6700114951466974ULL << 64) | 5150521797495201536ULL), // m=239
    (((__int128_t)11560969548788358ULL << 64) | 6528004383809233152ULL), // m=233
    (((__int128_t)7101110287921177ULL << 64) | 4964839118360008448ULL), // m=229
    (((__int128_t)8804211427022426ULL << 64) | 1720848776671001344ULL), // m=227
    (((__int128_t)9240461660392141ULL << 64) | 11161791116082051584ULL), // m=223
    (((__int128_t)7837025822737224ULL << 64) | 18050820367970017536ULL), // m=217
    (((__int128_t)7706892208217770ULL << 64) | 10896175520133382400ULL), // m=211
    (((__int128_t)6175506208042541ULL << 64) | 6042848966360184064ULL), // m=199
    (((__int128_t)11405197052119381ULL << 64) | 7978389471125905664ULL), // m=197
    (((__int128_t)10548166672197434ULL << 64) | 7075755759162582016ULL)  // m=193
};

// Helper to compute (X * y) % M without 128-bit overflow
unsigned __int128 mult_mod(unsigned __int128 X, uint32_t y, unsigned __int128 M) {
    unsigned __int128 result = 0;
    X = X % M;
    while (y > 0) {
        if (y & 1) {
            result = (result + X);
            if (result >= M) result -= M;
        }
        X = (X + X);
        if (X >= M) X -= M;
        y >>= 1;
    }
    return result;
}
