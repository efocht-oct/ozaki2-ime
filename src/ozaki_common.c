#include "ozaki_common.h"

// Pre-selected coprime moduli near 256. The FP64 basis is unchanged.
const int32_t MODULI_FP64[FP64_NUM_MODULI] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193
};

// CRT constants for the 15-modulus FP64 basis.
const __int128_t CRT_M_FP64 = (((__int128_t)12413391266671370ULL << 64) | 17550328509527814400ULL);
const __int128_t CRT_M_T_Y_T_FP64[FP64_NUM_MODULI] = {
    (((__int128_t)824326763802395ULL << 64) | 13415242489033580545ULL),
    (((__int128_t)8713713869545785ULL << 64) | 16298351871605898496ULL),
    (((__int128_t)9322309646907353ULL << 64) | 12523880316786161152ULL),
    (((__int128_t)9248223772380662ULL << 64) | 15133148467512226048ULL),
    (((__int128_t)8945682775172081ULL << 64) | 1669178525751606784ULL),
    (((__int128_t)6700114951466974ULL << 64) | 5150521797495201536ULL),
    (((__int128_t)11560969548788358ULL << 64) | 6528004383809233152ULL),
    (((__int128_t)7101110287921177ULL << 64) | 4964839118360008448ULL),
    (((__int128_t)8804211427022426ULL << 64) | 1720848776671001344ULL),
    (((__int128_t)9240461660392141ULL << 64) | 11161791116082051584ULL),
    (((__int128_t)7837025822737224ULL << 64) | 18050820367970017536ULL),
    (((__int128_t)7706892208217770ULL << 64) | 10896175520133382400ULL),
    (((__int128_t)6175506208042541ULL << 64) | 6042848966360184064ULL),
    (((__int128_t)11405197052119381ULL << 64) | 7978389471125905664ULL),
    (((__int128_t)10548166672197434ULL << 64) | 7075755759162582016ULL)
};

// The first seven moduli form the FP32 basis. Their product is 56 bits and
// covers the scaled FP32 accumulation range used by this implementation.
const int32_t MODULI_FP32[FP32_NUM_MODULI] = {
    256, 255, 253, 251, 247, 239, 233
};
const __int128_t CRT_M_FP32 = (__int128_t)57019730936213760ULL;
const __int128_t CRT_M_T_Y_T_FP32[FP32_NUM_MODULI] = {
    (__int128_t)46551264709643265ULL,
    (__int128_t)25267567042322176ULL,
    (__int128_t)26368808377616640ULL,
    (__int128_t)40209132970955520ULL,
    (__int128_t)19853023726778880ULL,
    (__int128_t)15746034484477440ULL,
    (__int128_t)54083092433061120ULL
};

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
