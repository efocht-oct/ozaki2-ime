#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <riscv_vector.h>
#include "ozaki_common.h"

/**
 * Inner Zvvm Kernel: Computes a single tile of C modulo m_t
 * Uses modern IME intrinsics with configurable lambda and dynamic geometry.
 */
#ifdef MOCK_IME
static inline void zvvm_kernel_int8_mac(
    int32_t* C_tile, const uint8_t* A_mod, const int8_t* B_mod, 
    int K, int lda, int ldb, size_t requested_vl, size_t lambda,
    size_t M_TILE, size_t K_EFF, size_t MAX_N_TILE) 
{
    (void)requested_vl;
    (void)lambda;
    (void)K_EFF;
    memset(C_tile, 0, M_TILE * MAX_N_TILE * sizeof(int32_t));
    for (int ii = 0; ii < (int)M_TILE; ii++) {
        for (int jj = 0; jj < (int)MAX_N_TILE; jj++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                uint8_t a = A_mod[ii * lda + k];
                int8_t b = B_mod[jj * ldb + k]; // B_mod column-major: B_mod[j*K + k] = B[k][j]
                acc += (int32_t)a * (int32_t)b;
            }
            C_tile[ii * MAX_N_TILE + jj] = acc;
        }
    }
}
#else
static inline void zvvm_kernel_int8_mac(
    int32_t* C_tile, const uint8_t* A_mod, const int8_t* B_mod, 
    int K, int lda, int ldb, size_t requested_vl, size_t lambda,
    size_t M_TILE, size_t K_EFF, size_t MAX_N_TILE) 
{
    (void)M_TILE;
    // Configure vector unit for SEW=32, LMUL=2 (EMUL_C=2)
    size_t vl = __riscv_vsetvl_e32m2(requested_vl);
    
    // Establish lambda (must be done after vsetvl as hardware may clamp lambda under WARL)
    __riscv_vsetlambda(lambda);
    
    // Zero out the INT32 accumulator
    vint32m2_t acc = __riscv_vmv_v_x_i32m2(0, vl);
    
    // Accumulate across the K dimension
    for (int k = 0; k < K; k += K_EFF) {
        const uint8_t* A_ptr = A_mod + k;
        // B_mod is column-major N×K: B_mod[j*K + k]. Pointer base &B_mod[j*K] + k offset.
        const int8_t* B_ptr = B_mod + k;

        // Order-preserving load for A (row-major) and for B (column-major = B^T row-major)
        vuint8m8_t va = __riscv_vmtl_v_u8m8(A_ptr, lda, vl);
        vint8m8_t vb = __riscv_vmtl_v_i8m8(B_ptr, ldb, vl);
        
        // Quad-widening MAC: INT8 x INT8 -> INT32 Accumulator (unsigned A x signed B)
        acc = __riscv_vqwmmacc_vv_i32m2_us_lm8(acc, va, vb, vl);
    }
    
    // Store the modular cross-term to memory
    __riscv_vmts_v_i32m2(C_tile, MAX_N_TILE, acc, vl);
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
    size_t MAX_N_TILE = (VLEN * 2 / 32) / lambda;

    if (M_TILE == 0) M_TILE = 1;
    if (MAX_N_TILE == 0) MAX_N_TILE = 1;

    // 3. Exponent Scaling & Modular Reduction (Software)
    uint8_t* A_mod[NUM_MODULI];
    int8_t* B_mod[NUM_MODULI];
    int32_t* C_mod[NUM_MODULI];
    for (int t = 0; t < NUM_MODULI; t++) {
        A_mod[t] = (uint8_t*)malloc(M * K * sizeof(uint8_t));
        B_mod[t] = (int8_t*)malloc(K * N * sizeof(int8_t));
        C_mod[t] = (int32_t*)malloc(M_TILE * MAX_N_TILE * sizeof(int32_t));
    }

    // Extract maximum exponents (D and E scale factors)
    double max_A = 0.0, max_B = 0.0;
    for (int i = 0; i < M * K; i++) max_A = fmax(max_A, fabs(A[i]));
    for (int i = 0; i < K * N; i++) max_B = fmax(max_B, fabs(B[i]));
    
    // Scale factors to map FP64 mantissas to integers
    double scale_A = (max_A == 0.0) ? 1.0 : exp2(-floor(log2(max_A)) + 52); 
    double scale_B = (max_B == 0.0) ? 1.0 : exp2(-floor(log2(max_B)) + 52);

    // Populate modulo matrices
    // A_mod mapped to positive modulo [0, MODULI[t]-1] (fits in uint8_t)
    for (int i = 0; i < M * K; i++) {
        int64_t a_int = (int64_t)round(A[i] * scale_A);
        for (int t = 0; t < NUM_MODULI; t++) {
            int64_t val = a_int % MODULI[t];
            if (val < 0) val += MODULI[t];
            A_mod[t][i] = (uint8_t)val;
        }
    }
    // B_mod stored column-major (N×K): B_mod[j*K + k] = B[k][j]
    // This makes B_mod a row-major B^T tile, correct for vmtl.v order-preserving load.
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            int64_t b_int = (int64_t)round(B[k * ldb + j] * scale_B);
            for (int t = 0; t < NUM_MODULI; t++) {
                int64_t val = b_int % MODULI[t];
                if (val < 0) val += MODULI[t];
                if (val > MODULI[t] / 2) val -= MODULI[t];
                B_mod[t][j * K + k] = (int8_t)val;
            }
        }
    }

    // 4. Tiled Matrix Multiplication (Zvvm Hardware)
    size_t vl = MAX_N_TILE * lambda; 

    for (size_t i = 0; i < (size_t)M; i += M_TILE) {
        for (size_t j = 0; j < (size_t)N; j += MAX_N_TILE) {
            
            // Compute this tile for all moduli
            for (int t = 0; t < NUM_MODULI; t++) {
                zvvm_kernel_int8_mac(
                    C_mod[t],
                    &A_mod[t][i * K],
                    &B_mod[t][j * K],  // column-major B: column j starts at j*K
                    K, K, K, vl, lambda,  // ldb=K (leading dim of column-major N×K)
                    M_TILE, K_EFF, MAX_N_TILE
                );
            }

            // 5. CRT Reconstruction & Inverse Scaling (Software)
            for (size_t ii = 0; ii < M_TILE && (i + ii) < (size_t)M; ii++) {
                for (size_t jj = 0; jj < MAX_N_TILE && (j + jj) < (size_t)N; jj++) {
                    unsigned __int128 exact_int_C_u = 0;
                    unsigned __int128 uCRT_M = (unsigned __int128)CRT_M;

                    // Reconstruct exact integer using Chinese Remainder Theorem
                    for (int t = 0; t < NUM_MODULI; t++) {
                        int32_t val = C_mod[t][ii * MAX_N_TILE + jj];
                        val %= MODULI[t];
                        if (val < 0) val += MODULI[t]; 
                        
                        unsigned __int128 term = mult_mod((unsigned __int128)CRT_M_t_y_t[t], (uint32_t)val, uCRT_M);
                        exact_int_C_u = (exact_int_C_u + term) % uCRT_M;
                    }
                    
                    __int128_t exact_int_C = (__int128_t)exact_int_C_u;
                    // Adjust for negative reconstructed values
                    if (exact_int_C > CRT_M / 2) {
                        exact_int_C -= CRT_M;
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
    for (int t = 0; t < NUM_MODULI; t++) {
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
