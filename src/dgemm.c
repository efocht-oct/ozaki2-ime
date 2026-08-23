#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <riscv_vector.h>
#include "ozaki_common.h"

typedef int8_t ozaki_mod_t;

/**
 * Inner Zvvm Kernel: Computes a single tile of C modulo m_t
 * Uses modern IME intrinsics with configurable lambda and dynamic geometry.
 */
#ifdef MOCK_IME
static inline void zvvm_kernel_int8_mac(
    int32_t* C_tile, const ozaki_mod_t* A_mod, const ozaki_mod_t* B_mod,
    int K, int lda, int ldb, size_t requested_vl, size_t lambda,
    size_t M_TILE, size_t K_EFF, size_t MAX_N_TILE,
    size_t A_PANEL_STRIDE, size_t B_PANEL_STRIDE)
{
    (void)requested_vl;
    (void)lambda;
    (void)K_EFF;
    (void)A_PANEL_STRIDE;
    (void)B_PANEL_STRIDE;
    memset(C_tile, 0, M_TILE * MAX_N_TILE * sizeof(int32_t));
    for (int ii = 0; ii < (int)M_TILE; ii++) {
        for (int jj = 0; jj < (int)MAX_N_TILE; jj++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int8_t a = A_mod[ii * lda + k];
                int8_t b = B_mod[jj * ldb + k]; // B_mod column-major: B_mod[j*K + k] = B[k][j]
                acc += (int32_t)a * (int32_t)b;
            }
            C_tile[ii * MAX_N_TILE + jj] = acc;
        }
    }
}
#else
#define DEFINE_ZVVM_KERNEL(CSUFFIX, ISUFFIX, CVTYPE, IVTYPE, CVSETVL, IVSETVL, VMV, VQWMMACC, VMTS) \
static inline void zvvm_kernel_int8_mac_##CSUFFIX##_##ISUFFIX( \
    int32_t* C_tile, const ozaki_mod_t* A_mod, const ozaki_mod_t* B_mod, \
    int K, int lda, int ldb, size_t requested_vl, size_t lambda, \
    size_t M_TILE, size_t K_EFF, size_t N_TILE, size_t C_LD, \
    size_t A_PANEL_STRIDE, size_t B_PANEL_STRIDE) \
{ \
    size_t vl = CVSETVL(requested_vl); \
    __riscv_vsetlambda(lambda); \
    CVTYPE acc = VMV(0, vl); \
    for (size_t k0 = 0, panel = 0; k0 < (size_t)K; k0 += K_EFF, panel++) { \
        size_t vl_a = IVSETVL(M_TILE * K_EFF); \
        __riscv_vsetlambda(lambda); \
        IVTYPE va = __riscv_vmtl_v_i8##ISUFFIX(A_mod + panel * A_PANEL_STRIDE, lda, vl_a); \
        size_t vl_b = IVSETVL(N_TILE * K_EFF); \
        __riscv_vsetlambda(lambda); \
        IVTYPE vb = __riscv_vmtl_v_i8##ISUFFIX(B_mod + panel * B_PANEL_STRIDE, ldb, vl_b); \
        __riscv_vsetlambda(lambda); \
        acc = VQWMMACC(acc, va, vb, vl); \
    } \
    VMTS(C_tile, C_LD, acc, vl); \
}

DEFINE_ZVVM_KERNEL(m1, m1, vint32m1_t, vint8m1_t, __riscv_vsetvl_e32m1, __riscv_vsetvl_e8m1, __riscv_vmv_v_x_i32m1, __riscv_vqwmmacc_vv_i32m1, __riscv_vmts_v_i32m1)
DEFINE_ZVVM_KERNEL(m1, m2, vint32m1_t, vint8m2_t, __riscv_vsetvl_e32m1, __riscv_vsetvl_e8m2, __riscv_vmv_v_x_i32m1, __riscv_vqwmmacc_vv_i32m1_lm2, __riscv_vmts_v_i32m1)
DEFINE_ZVVM_KERNEL(m1, m4, vint32m1_t, vint8m4_t, __riscv_vsetvl_e32m1, __riscv_vsetvl_e8m4, __riscv_vmv_v_x_i32m1, __riscv_vqwmmacc_vv_i32m1_lm4, __riscv_vmts_v_i32m1)
DEFINE_ZVVM_KERNEL(m1, m8, vint32m1_t, vint8m8_t, __riscv_vsetvl_e32m1, __riscv_vsetvl_e8m8, __riscv_vmv_v_x_i32m1, __riscv_vqwmmacc_vv_i32m1_lm8, __riscv_vmts_v_i32m1)
DEFINE_ZVVM_KERNEL(m2, m1, vint32m2_t, vint8m1_t, __riscv_vsetvl_e32m2, __riscv_vsetvl_e8m1, __riscv_vmv_v_x_i32m2, __riscv_vqwmmacc_vv_i32m2, __riscv_vmts_v_i32m2)
DEFINE_ZVVM_KERNEL(m2, m2, vint32m2_t, vint8m2_t, __riscv_vsetvl_e32m2, __riscv_vsetvl_e8m2, __riscv_vmv_v_x_i32m2, __riscv_vqwmmacc_vv_i32m2_lm2, __riscv_vmts_v_i32m2)
DEFINE_ZVVM_KERNEL(m2, m4, vint32m2_t, vint8m4_t, __riscv_vsetvl_e32m2, __riscv_vsetvl_e8m4, __riscv_vmv_v_x_i32m2, __riscv_vqwmmacc_vv_i32m2_lm4, __riscv_vmts_v_i32m2)
DEFINE_ZVVM_KERNEL(m2, m8, vint32m2_t, vint8m8_t, __riscv_vsetvl_e32m2, __riscv_vsetvl_e8m8, __riscv_vmv_v_x_i32m2, __riscv_vqwmmacc_vv_i32m2_lm8, __riscv_vmts_v_i32m2)
DEFINE_ZVVM_KERNEL(m4, m1, vint32m4_t, vint8m1_t, __riscv_vsetvl_e32m4, __riscv_vsetvl_e8m1, __riscv_vmv_v_x_i32m4, __riscv_vqwmmacc_vv_i32m4, __riscv_vmts_v_i32m4)
DEFINE_ZVVM_KERNEL(m4, m2, vint32m4_t, vint8m2_t, __riscv_vsetvl_e32m4, __riscv_vsetvl_e8m2, __riscv_vmv_v_x_i32m4, __riscv_vqwmmacc_vv_i32m4_lm2, __riscv_vmts_v_i32m4)
DEFINE_ZVVM_KERNEL(m4, m4, vint32m4_t, vint8m4_t, __riscv_vsetvl_e32m4, __riscv_vsetvl_e8m4, __riscv_vmv_v_x_i32m4, __riscv_vqwmmacc_vv_i32m4_lm4, __riscv_vmts_v_i32m4)
DEFINE_ZVVM_KERNEL(m4, m8, vint32m4_t, vint8m8_t, __riscv_vsetvl_e32m4, __riscv_vsetvl_e8m8, __riscv_vmv_v_x_i32m4, __riscv_vqwmmacc_vv_i32m4_lm8, __riscv_vmts_v_i32m4)
DEFINE_ZVVM_KERNEL(m8, m1, vint32m8_t, vint8m1_t, __riscv_vsetvl_e32m8, __riscv_vsetvl_e8m1, __riscv_vmv_v_x_i32m8, __riscv_vqwmmacc_vv_i32m8, __riscv_vmts_v_i32m8)
DEFINE_ZVVM_KERNEL(m8, m2, vint32m8_t, vint8m2_t, __riscv_vsetvl_e32m8, __riscv_vsetvl_e8m2, __riscv_vmv_v_x_i32m8, __riscv_vqwmmacc_vv_i32m8_lm2, __riscv_vmts_v_i32m8)
DEFINE_ZVVM_KERNEL(m8, m4, vint32m8_t, vint8m4_t, __riscv_vsetvl_e32m8, __riscv_vsetvl_e8m4, __riscv_vmv_v_x_i32m8, __riscv_vqwmmacc_vv_i32m8_lm4, __riscv_vmts_v_i32m8)
DEFINE_ZVVM_KERNEL(m8, m8, vint32m8_t, vint8m8_t, __riscv_vsetvl_e32m8, __riscv_vsetvl_e8m8, __riscv_vmv_v_x_i32m8, __riscv_vqwmmacc_vv_i32m8_lm8, __riscv_vmts_v_i32m8)

#undef DEFINE_ZVVM_KERNEL

static inline void zvvm_kernel_int8_mac(
    int32_t* C_tile, const ozaki_mod_t* A_mod, const ozaki_mod_t* B_mod,
    int K, int lda, int ldb, size_t requested_vl, size_t lambda,
    size_t M_TILE, size_t K_EFF, size_t MAX_N_TILE,
    size_t A_PANEL_STRIDE, size_t B_PANEL_STRIDE)
{
    (void)M_TILE;
    size_t vlmax_e32m1 = __riscv_vsetvl_e32m1(~0ULL);
    size_t emul_c = vlmax_e32m1 / (lambda * lambda);
    size_t input_lmul = K_EFF / (lambda * 4);

#define DISPATCH_INPUT(CSUFFIX) \
    switch (input_lmul) { \
    case 1: zvvm_kernel_int8_mac_##CSUFFIX##_m1(C_tile, A_mod, B_mod, K, lda, ldb, requested_vl, lambda, M_TILE, K_EFF, MAX_N_TILE, MAX_N_TILE, A_PANEL_STRIDE, B_PANEL_STRIDE); break; \
    case 2: zvvm_kernel_int8_mac_##CSUFFIX##_m2(C_tile, A_mod, B_mod, K, lda, ldb, requested_vl, lambda, M_TILE, K_EFF, MAX_N_TILE, MAX_N_TILE, A_PANEL_STRIDE, B_PANEL_STRIDE); break; \
    case 4: zvvm_kernel_int8_mac_##CSUFFIX##_m4(C_tile, A_mod, B_mod, K, lda, ldb, requested_vl, lambda, M_TILE, K_EFF, MAX_N_TILE, MAX_N_TILE, A_PANEL_STRIDE, B_PANEL_STRIDE); break; \
    case 8: zvvm_kernel_int8_mac_##CSUFFIX##_m8(C_tile, A_mod, B_mod, K, lda, ldb, requested_vl, lambda, M_TILE, K_EFF, MAX_N_TILE, MAX_N_TILE, A_PANEL_STRIDE, B_PANEL_STRIDE); break; \
    default: memset(C_tile, 0, M_TILE * MAX_N_TILE * sizeof(int32_t)); break; \
    }

#define DISPATCH_M8_CHUNK(N_OFFSET, N_TILE) \
    switch (input_lmul) { \
    case 1: zvvm_kernel_int8_mac_m8_m1(C_tile + (N_OFFSET), A_mod, B_mod + 4 * (N_OFFSET) * ldb, K, lda, ldb, M_TILE * (N_TILE), lambda, M_TILE, K_EFF, (N_TILE), MAX_N_TILE, A_PANEL_STRIDE, B_PANEL_STRIDE); break; \
    case 2: zvvm_kernel_int8_mac_m8_m2(C_tile + (N_OFFSET), A_mod, B_mod + 4 * (N_OFFSET) * ldb, K, lda, ldb, M_TILE * (N_TILE), lambda, M_TILE, K_EFF, (N_TILE), MAX_N_TILE, A_PANEL_STRIDE, B_PANEL_STRIDE); break; \
    case 4: zvvm_kernel_int8_mac_m8_m4(C_tile + (N_OFFSET), A_mod, B_mod + 4 * (N_OFFSET) * ldb, K, lda, ldb, M_TILE * (N_TILE), lambda, M_TILE, K_EFF, (N_TILE), MAX_N_TILE, A_PANEL_STRIDE, B_PANEL_STRIDE); break; \
    case 8: zvvm_kernel_int8_mac_m8_m8(C_tile + (N_OFFSET), A_mod, B_mod + 4 * (N_OFFSET) * ldb, K, lda, ldb, M_TILE * (N_TILE), lambda, M_TILE, K_EFF, (N_TILE), MAX_N_TILE, A_PANEL_STRIDE, B_PANEL_STRIDE); break; \
    default: memset(C_tile, 0, M_TILE * MAX_N_TILE * sizeof(int32_t)); break; \
    }

    switch (emul_c) {
    case 1:
        DISPATCH_INPUT(m1);
        break;
    case 2:
        DISPATCH_INPUT(m2);
        break;
    case 4:
        DISPATCH_INPUT(m4);
        break;
    case 8:
        DISPATCH_INPUT(m8);
        break;
    default:
        if (emul_c > 8 && (emul_c % 8) == 0) {
            for (size_t n0 = 0; n0 < MAX_N_TILE; n0 += lambda * 8) {
                size_t n_tile = lambda * 8;
                if (n_tile > MAX_N_TILE - n0) n_tile = MAX_N_TILE - n0;
                DISPATCH_M8_CHUNK(n0, n_tile);
            }
        } else {
            memset(C_tile, 0, M_TILE * MAX_N_TILE * sizeof(int32_t));
        }
        break;
    }
#undef DISPATCH_INPUT
#undef DISPATCH_M8_CHUNK
}
#endif

/**
 * Dynamic DGEMM utilizing Ozaki Scheme II with dynamic lambda/VLEN detection.
 */
void ozaki_dgemm(int M, int N, int K, double alpha, 
                 const double *A, int lda, 
                 const double *B, int ldb, 
                 double beta, double *C, int ldc) 
{
    // 1. Detect dynamic lambda and VLEN from the hardware CSR
    size_t lambda = __riscv_vsetlambda(0);
    if (lambda == 0) lambda = 8; // Fallback to 8 if unconfigured/unsupported

    size_t vlmax_e32m1 = __riscv_vsetvl_e32m1(~0ULL);
    size_t VLEN = vlmax_e32m1 * 32;

    // 2. Compute runtime configuration parameters from (VLEN, lambda, SEW=32)
    size_t M_TILE = VLEN / (32 * lambda);
    size_t K_EFF = lambda * 4;
    size_t MAX_N_TILE = M_TILE;
#ifndef MOCK_IME
    size_t emul_c = vlmax_e32m1 / (lambda * lambda);
    size_t max_input_lmul = emul_c < 8 ? emul_c : 8;
    size_t input_lmul = 1;
    while (input_lmul < max_input_lmul && K_EFF < (size_t)K) {
        input_lmul *= 2;
        K_EFF = lambda * 4 * input_lmul;
    }
    size_t input_linesize = lambda * input_lmul;
    size_t k_panels = ((size_t)K + K_EFF - 1) / K_EFF;
    size_t a_panel_stride = 4 * M * input_linesize;
    size_t b_panel_stride = 4 * N * input_linesize;
#endif

    if (M_TILE == 0) M_TILE = 1;
    if (MAX_N_TILE == 0) MAX_N_TILE = 1;

    // 3. Exponent Scaling & Modular Reduction (Software)
    ozaki_mod_t* A_mod[FP64_NUM_MODULI];
    ozaki_mod_t* B_mod[FP64_NUM_MODULI];
    int32_t* C_mod[FP64_NUM_MODULI];
    for (int t = 0; t < FP64_NUM_MODULI; t++) {
#ifdef MOCK_IME
        A_mod[t] = (ozaki_mod_t*)malloc(M * K * sizeof(ozaki_mod_t));
        B_mod[t] = (ozaki_mod_t*)malloc(K * N * sizeof(ozaki_mod_t));
#else
    A_mod[t] = (ozaki_mod_t*)calloc(k_panels * a_panel_stride, sizeof(ozaki_mod_t));
    B_mod[t] = (ozaki_mod_t*)calloc(k_panels * b_panel_stride, sizeof(ozaki_mod_t));
#endif
        C_mod[t] = (int32_t*)malloc(M_TILE * MAX_N_TILE * sizeof(int32_t));
    }

    // Extract maximum exponents (D and E scale factors)
    double max_A = 0.0, max_B = 0.0;
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) max_A = fmax(max_A, fabs(A[i * lda + k]));
    }
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) max_B = fmax(max_B, fabs(B[k * ldb + j]));
    }
    
    // Scale factors to map FP64 mantissas to integers
    double scale_A = (max_A == 0.0) ? 1.0 : exp2(-floor(log2(max_A)) + 52); 
    double scale_B = (max_B == 0.0) ? 1.0 : exp2(-floor(log2(max_B)) + 52);

    // Populate modulo matrices
    // A_mod mapped to centered signed modulo [-MODULI_FP64[t]/2, MODULI_FP64[t]/2) (fits in int8_t)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            int64_t a_int = (int64_t)round(A[i * lda + k] * scale_A);
            for (int t = 0; t < FP64_NUM_MODULI; t++) {
                int64_t val = a_int % MODULI_FP64[t];
                if (val < 0) val += MODULI_FP64[t];
                if (val > MODULI_FP64[t] / 2) val -= MODULI_FP64[t];
#ifdef MOCK_IME
                A_mod[t][i * K + k] = (int8_t)val;
#else
                size_t panel = (size_t)k / K_EFF;
                size_t kk = (size_t)k % K_EFF;
                size_t tr = 4 * (size_t)i + kk / input_linesize;
                size_t tc = kk % input_linesize;
                A_mod[t][panel * a_panel_stride + tr * input_linesize + tc] = (int8_t)val;
#endif
            }
        }
    }
    // B_mod stored column-major (N×K): B_mod[j*K + k] = B[k][j]
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            int64_t b_int = (int64_t)round(B[k * ldb + j] * scale_B);
            for (int t = 0; t < FP64_NUM_MODULI; t++) {
                int64_t val = b_int % MODULI_FP64[t];
                if (val < 0) val += MODULI_FP64[t];
                if (val > MODULI_FP64[t] / 2) val -= MODULI_FP64[t];
#ifdef MOCK_IME
                B_mod[t][j * K + k] = (int8_t)val;
#else
                size_t panel = (size_t)k / K_EFF;
                size_t kk = (size_t)k % K_EFF;
                size_t tr = 4 * (size_t)j + kk / input_linesize;
                size_t tc = kk % input_linesize;
                B_mod[t][panel * b_panel_stride + tr * input_linesize + tc] = (int8_t)val;
#endif
            }
        }
    }

    // 4. Tiled Matrix Multiplication (Zvvm Hardware)
    size_t vl = M_TILE * MAX_N_TILE;

    for (size_t i = 0; i < (size_t)M; i += M_TILE) {
        for (size_t j = 0; j < (size_t)N; j += MAX_N_TILE) {
            
            // Compute this tile for all moduli
            for (int t = 0; t < FP64_NUM_MODULI; t++) {
                zvvm_kernel_int8_mac(
                    C_mod[t],
#ifdef MOCK_IME
                    &A_mod[t][i * K],
                    &B_mod[t][j * K],  // column-major B: column j starts at j*K
                    K, K, K, vl, lambda,  // ldb=K (leading dim of column-major N×K)
                    M_TILE, K_EFF, MAX_N_TILE, 0, 0
#else
                    &A_mod[t][4 * i * input_linesize],
                    &B_mod[t][4 * j * input_linesize],
                    K, (int)input_linesize, (int)input_linesize, vl, lambda,
                    M_TILE, K_EFF, MAX_N_TILE, a_panel_stride, b_panel_stride
#endif
                );
            }

            // 5. CRT Reconstruction & Inverse Scaling (Software)
            for (size_t ii = 0; ii < M_TILE && (i + ii) < (size_t)M; ii++) {
                for (size_t jj = 0; jj < MAX_N_TILE && (j + jj) < (size_t)N; jj++) {
                    unsigned __int128 exact_int_C_u = 0;
                    unsigned __int128 uCRT_M_FP64 = (unsigned __int128)CRT_M_FP64;

                    // Reconstruct exact integer using Chinese Remainder Theorem
                    for (int t = 0; t < FP64_NUM_MODULI; t++) {
                        int32_t val = C_mod[t][ii * MAX_N_TILE + jj];
                        val %= MODULI_FP64[t];
                        if (val < 0) val += MODULI_FP64[t];
                        
                        unsigned __int128 term = mult_mod((unsigned __int128)CRT_M_T_Y_T_FP64[t], (uint32_t)val, uCRT_M_FP64);
                        exact_int_C_u = (exact_int_C_u + term) % uCRT_M_FP64;
                    }
                    
                    __int128_t exact_int_C = (__int128_t)exact_int_C_u;
                    // Adjust for negative reconstructed values
                    if (exact_int_C > (__int128_t)(uCRT_M_FP64 / 2)) {
                        exact_int_C -= uCRT_M_FP64;
                    }

                    // Convert to FP64, apply inverse scaling, and accumulate
                    double C_exact_fp = (double)exact_int_C;
                    C_exact_fp /= (scale_A * scale_B);

                    int c_idx = (i + ii) * ldc + (j + jj);
                    if (beta == 0.0) {
                        C[c_idx] = alpha * C_exact_fp;
                    } else {
                        C[c_idx] = alpha * C_exact_fp + beta * C[c_idx];
                    }
                }
            }
        }
    }

    // Cleanup
    for (int t = 0; t < FP64_NUM_MODULI; t++) {
        free(A_mod[t]);
        free(B_mod[t]);
        free(C_mod[t]);
    }
}

/**
 * Legacy/Fixed DGEMM Wrapper for lambda=8 (does not need lambda as input parameter)
 */
void ozaki_dgemm_l8(int M, int N, int K, double alpha, 
                    const double *A, int lda, 
                    const double *B, int ldb, 
                    double beta, double *C, int ldc) 
{
    // In real HW / correct test execution, lambda=8 must be set in the test harness
    // before this function is called.
    ozaki_dgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
