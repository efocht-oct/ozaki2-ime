#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <riscv_vector.h>
#include "ozaki_common.h"

// Hardware Parameters
#define M_TILE 64
#define K_EFF 32
#define MAX_N_TILE 64

/**
 * Inner Zvvm Kernel: Computes a single 64x64 tile of C modulo m_t
 * Uses modern IME intrinsics with configurable lambda.
 */
#ifdef MOCK_IME
static inline void zvvm_kernel_int8_mac(
    int32_t* C_tile, const uint8_t* A_mod, const int8_t* B_mod, 
    int K, int lda, int ldb, size_t requested_vl, size_t lambda) 
{
    (void)requested_vl;
    (void)lambda;
    memset(C_tile, 0, M_TILE * MAX_N_TILE * sizeof(int32_t));
    for (int ii = 0; ii < M_TILE; ii++) {
        for (int jj = 0; jj < MAX_N_TILE; jj++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                uint8_t a = A_mod[ii * lda + k];
                int8_t b = B_mod[k * ldb + jj];
                acc += (int32_t)a * (int32_t)b;
            }
            C_tile[ii * MAX_N_TILE + jj] = acc;
        }
    }
}
#else
static inline void zvvm_kernel_int8_mac(
    int32_t* C_tile, const uint8_t* A_mod, const int8_t* B_mod, 
    int K, int lda, int ldb, size_t requested_vl, size_t lambda) 
{
    // Configure vector unit for SEW=32, LMUL=2 (EMUL_C=2 for VLEN=16384, lambda=8)
    // Note: For LMUL=8 inputs, EMUL_C = LMUL / 4 = 2.
    size_t vl = __riscv_vsetvl_e32m2(requested_vl);
    
    // Establish lambda (must be done after vsetvl as hardware may clamp lambda under WARL)
    __riscv_vsetlambda(lambda);
    
    // Zero out the INT32 accumulator
    vint32m2_t acc = __riscv_vmv_v_x_i32m2(0, vl);
    
    // Accumulate across the K dimension
    for (int k = 0; k < K; k += K_EFF) {
        const uint8_t* A_ptr = A_mod + k;
        const int8_t* B_ptr = B_mod + k; // B is assumed column-major in memory layout for transposing load
        
        // Order-preserving load for A (Row-major) - cast to unsigned for _us intrinsic
        vuint8m8_t va = __riscv_vmtl_v_u8m8(A_ptr, lda, vl);
        
        // Transposing load for B (Column-major to row-major)
        vint8m8_t vb = __riscv_vmttl_v_i8m8(B_ptr, ldb, vl);
        
        // Quad-widening MAC: INT8 x INT8 -> INT32 Accumulator (unsigned A x signed B)
        acc = __riscv_vqwmmacc_vv_i32m2_us_lm8(acc, va, vb, vl);
    }
    
    // Store the modular cross-term 64x64 tile to memory
    __riscv_vmts_v_i32m2(C_tile, MAX_N_TILE, acc, vl);
}
#endif

/**
 * Classic DGEMM Wrapper utilizing Ozaki Scheme II on Zvvm
 */
void ozaki_dgemm(int M, int N, int K, double alpha, 
                 const double *A, int lda, 
                 const double *B, int ldb, 
                 double beta, double *C, int ldc, size_t lambda) 
{
    // 1. Exponent Scaling & Modular Reduction (Software)
    uint8_t* A_mod[NUM_MODULI];
    int8_t* B_mod[NUM_MODULI];
    for (int t = 0; t < NUM_MODULI; t++) {
        A_mod[t] = (uint8_t*)malloc(M * K * sizeof(uint8_t));
        B_mod[t] = (int8_t*)malloc(K * N * sizeof(int8_t));
    }

    // Extract maximum exponents (D and E scale factors)
    double max_A = 0.0, max_B = 0.0;
    for (int i = 0; i < M * K; i++) max_A = fmax(max_A, fabs(A[i]));
    for (int i = 0; i < K * N; i++) max_B = fmax(max_B, fabs(B[i]));
    
    // Scale factors to map FP64 mantissas to integers
    double scale_A = (max_A == 0.0) ? 1.0 : exp2(-floor(log2(max_A)) + 52); 
    double scale_B = (max_B == 0.0) ? 1.0 : exp2(-floor(log2(max_B)) + 52);

    // Populate modulo matrices
    // A_mod is mapped to positive modulo [0, MODULI[t]-1] (fits in uint8_t up to 255)
    for (int i = 0; i < M * K; i++) {
        int64_t a_int = (int64_t)round(A[i] * scale_A);
        for (int t = 0; t < NUM_MODULI; t++) {
            int64_t val = a_int % MODULI[t];
            if (val < 0) val += MODULI[t];
            A_mod[t][i] = (uint8_t)val;
        }
    }
    // B_mod is mapped to symmetric modulo [-MODULI[t]/2, MODULI[t]/2] (fits in int8_t)
    for (int i = 0; i < K * N; i++) {
        int64_t b_int = (int64_t)round(B[i] * scale_B);
        for (int t = 0; t < NUM_MODULI; t++) {
            int64_t val = b_int % MODULI[t];
            if (val < 0) val += MODULI[t];
            if (val > MODULI[t] / 2) val -= MODULI[t];
            B_mod[t][i] = (int8_t)val;
        }
    }

    // 2. Tiled Matrix Multiplication (Zvvm Hardware)
    int32_t C_mod[NUM_MODULI][M_TILE * MAX_N_TILE];
    
    // VL required to process full 64x64 block. N_TILE * lambda = 64 * 8 = 512.
    size_t vl = MAX_N_TILE * 8; 

    for (int i = 0; i < M; i += M_TILE) {
        for (int j = 0; j < N; j += MAX_N_TILE) {
            
            // Compute this tile for all moduli
            for (int t = 0; t < NUM_MODULI; t++) {
                zvvm_kernel_int8_mac(
                    C_mod[t], 
                    &A_mod[t][i * K], 
                    &B_mod[t][j],
                    K, K, N, vl, lambda
                );
            }

            // 3. CRT Reconstruction & Inverse Scaling (Software)
            for (int ii = 0; ii < M_TILE && (i + ii) < M; ii++) {
                for (int jj = 0; jj < MAX_N_TILE && (j + jj) < N; jj++) {
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
    }
}
