#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>

// --- Hardware Parameters ---
// Based on VLEN=16384, lambda=8, W=4 (INT8 -> INT32 quad-widening)
#define M_TILE 64
#define K_EFF 32
#define MAX_N_TILE 64

// --- Ozaki Scheme II Parameters ---
#define NUM_MODULI 15
const int32_t MODULI[NUM_MODULI] = {
    256, 255, 253, 251, 247, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193
};

__int128_t CRT_M; 
__int128_t CRT_M_t_y_t[NUM_MODULI];

// Extended Euclidean Algorithm for Modular Inverse
__int128_t mod_inverse(__int128_t a, __int128_t m) {
    __int128_t m0 = m, t, q;
    __int128_t x0 = 0, x1 = 1;
    if (m == 1) return 0;
    while (a > 1) {
        q = a / m;
        t = m;
        m = a % m, a = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }
    if (x1 < 0) x1 += m0;
    return x1;
}

// Initialize Chinese Remainder Theorem constants dynamically
void init_crt() {
    unsigned __int128 uCRT_M = 1;
    for (int i = 0; i < NUM_MODULI; i++) {
        uCRT_M *= MODULI[i];
    }
    CRT_M = (__int128_t)uCRT_M;

    for (int i = 0; i < NUM_MODULI; i++) {
        unsigned __int128 M_t = uCRT_M / MODULI[i];
        __int128_t rem = (__int128_t)(M_t % MODULI[i]);
        __int128_t y_t = mod_inverse(rem, MODULI[i]);
        CRT_M_t_y_t[i] = (__int128_t)(M_t * y_t);
    }
}

// --- Zvvm Hardware Mock ---
/**
 * Software mock of the Zvvm vqwmmacc.vv inner kernel.
 * Replaces the inline assembly so the test can be natively compiled and run.
 */
static inline void zvvm_kernel_int8_mac(
    int32_t* C_tile, const int8_t* A_mod, const int8_t* B_mod, 
    int K, int lda, int ldb) 
{
    // C_tile is M_TILE x MAX_N_TILE (64x64)
    memset(C_tile, 0, M_TILE * MAX_N_TILE * sizeof(int32_t));

    // Simulate the hardware MAC operations
    for (int ii = 0; ii < M_TILE; ii++) {
        for (int jj = 0; jj < MAX_N_TILE; jj++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int8_t a = A_mod[ii * lda + k];
                int8_t b = B_mod[k * ldb + jj];
                acc += (int32_t)a * (int32_t)b;
            }
            C_tile[ii * MAX_N_TILE + jj] = acc;
        }
    }
}

// ============================================================================
// ==================== DOUBLE PRECISION GEMM (FP64) ==========================
// ============================================================================

void ozaki_dgemm(int M, int N, int K, double alpha, 
                 const double *A, int lda, 
                 const double *B, int ldb, 
                 double beta, double *C, int ldc) 
{
    int8_t* A_mod[NUM_MODULI];
    int8_t* B_mod[NUM_MODULI];
    for (int t = 0; t < NUM_MODULI; t++) {
        A_mod[t] = (int8_t*)malloc(M * K * sizeof(int8_t));
        B_mod[t] = (int8_t*)malloc(K * N * sizeof(int8_t));
    }

    double max_A = 0.0, max_B = 0.0;
    for (int i = 0; i < M * K; i++) max_A = fmax(max_A, fabs(A[i]));
    for (int i = 0; i < K * N; i++) max_B = fmax(max_B, fabs(B[i]));
    
    double scale_A = (max_A == 0.0) ? 1.0 : exp2(-floor(log2(max_A)) + 52); 
    double scale_B = (max_B == 0.0) ? 1.0 : exp2(-floor(log2(max_B)) + 52);

    for (int i = 0; i < M * K; i++) {
        int64_t a_int = (int64_t)round(A[i] * scale_A);
        for (int t = 0; t < NUM_MODULI; t++) {
            int64_t r = a_int % MODULI[t];
            if (r < 0) r += MODULI[t];
            if (r > MODULI[t] / 2) r -= MODULI[t];
            A_mod[t][i] = (int8_t)r; 
        }
    }
    for (int i = 0; i < K * N; i++) {
        int64_t b_int = (int64_t)round(B[i] * scale_B);
        for (int t = 0; t < NUM_MODULI; t++) {
            int64_t r = b_int % MODULI[t];
            if (r < 0) r += MODULI[t];
            if (r > MODULI[t] / 2) r -= MODULI[t];
            B_mod[t][i] = (int8_t)r;
        }
    }

    int32_t C_mod[NUM_MODULI][M_TILE * MAX_N_TILE];
    
    for (int i = 0; i < M; i += M_TILE) {
        for (int j = 0; j < N; j += MAX_N_TILE) {
            
            for (int t = 0; t < NUM_MODULI; t++) {
                zvvm_kernel_int8_mac(
                    C_mod[t], &A_mod[t][i * lda], &B_mod[t][j], K, lda, ldb
                );
            }

            for (int ii = 0; ii < M_TILE && (i + ii) < M; ii++) {
                for (int jj = 0; jj < MAX_N_TILE && (j + jj) < N; jj++) {
                    unsigned __int128 exact_int_C_u = 0;
                    unsigned __int128 uCRT_M = (unsigned __int128)CRT_M;

                    for (int t = 0; t < NUM_MODULI; t++) {
                        int64_t val = C_mod[t][ii * MAX_N_TILE + jj];
                        val %= MODULI[t];
                        if (val < 0) val += MODULI[t]; 
                        
                        unsigned __int128 term = (unsigned __int128)val * (unsigned __int128)CRT_M_t_y_t[t];
                        exact_int_C_u = (exact_int_C_u + (term % uCRT_M)) % uCRT_M;
                    }
                    
                    __int128_t exact_int_C = (__int128_t)exact_int_C_u;
                    if (exact_int_C > CRT_M / 2) exact_int_C -= CRT_M;

                    double C_exact_fp = (double)exact_int_C;
                    C_exact_fp /= (scale_A * scale_B);

                    int c_idx = (i + ii) * ldc + (j + jj);
                    if (beta == 0.0) C[c_idx] = alpha * C_exact_fp;
                    else C[c_idx] = alpha * C_exact_fp + beta * C[c_idx];
                }
            }
        }
    }

    for (int t = 0; t < NUM_MODULI; t++) {
        free(A_mod[t]); free(B_mod[t]);
    }
}

void reference_dgemm(int M, int N, int K, double alpha, 
                     const double *A, int lda, 
                     const double *B, int ldb, 
                     double beta, double *C, int ldc) 
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) sum += A[i * lda + k] * B[k * ldb + j];
            if (beta == 0.0) C[i * ldc + j] = alpha * sum;
            else C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}


// ============================================================================
// ==================== 54-BIT UNSIGNED INTEGER GEMM ==========================
// ============================================================================

void ozaki_u54_gemm(int M, int N, int K, 
                    const uint64_t *A, int lda, 
                    const uint64_t *B, int ldb, 
                    unsigned __int128 *C, int ldc) 
{
    int8_t* A_mod[NUM_MODULI];
    int8_t* B_mod[NUM_MODULI];
    for (int t = 0; t < NUM_MODULI; t++) {
        A_mod[t] = (int8_t*)malloc(M * K * sizeof(int8_t));
        B_mod[t] = (int8_t*)malloc(K * N * sizeof(int8_t));
    }

    // No Exponent Scaling needed for pure integers!
    // Just map straight to the symmetric modulo representation.
    for (int i = 0; i < M * K; i++) {
        for (int t = 0; t < NUM_MODULI; t++) {
            int64_t r = A[i] % MODULI[t];
            // Since A is unsigned, r is always positive. Just center it:
            if (r > MODULI[t] / 2) r -= MODULI[t];   
            A_mod[t][i] = (int8_t)r; 
        }
    }
    for (int i = 0; i < K * N; i++) {
        for (int t = 0; t < NUM_MODULI; t++) {
            int64_t r = B[i] % MODULI[t];
            if (r > MODULI[t] / 2) r -= MODULI[t];
            B_mod[t][i] = (int8_t)r;
        }
    }

    int32_t C_mod[NUM_MODULI][M_TILE * MAX_N_TILE];
    
    // Tiled Matrix Multiplication & CRT Reconstruction
    for (int i = 0; i < M; i += M_TILE) {
        for (int j = 0; j < N; j += MAX_N_TILE) {
            
            for (int t = 0; t < NUM_MODULI; t++) {
                zvvm_kernel_int8_mac(
                    C_mod[t], &A_mod[t][i * lda], &B_mod[t][j], K, lda, ldb
                );
            }

            for (int ii = 0; ii < M_TILE && (i + ii) < M; ii++) {
                for (int jj = 0; jj < MAX_N_TILE && (j + jj) < N; jj++) {
                    unsigned __int128 exact_int_C_u = 0;
                    unsigned __int128 uCRT_M = (unsigned __int128)CRT_M;

                    for (int t = 0; t < NUM_MODULI; t++) {
                        int64_t val = C_mod[t][ii * MAX_N_TILE + jj];
                        val %= MODULI[t];
                        if (val < 0) val += MODULI[t]; // Ensure positive remainder
                        
                        unsigned __int128 term = (unsigned __int128)val * (unsigned __int128)CRT_M_t_y_t[t];
                        exact_int_C_u = (exact_int_C_u + (term % uCRT_M)) % uCRT_M;
                    }
                    
                    // The result is strictly positive, so no negative centering is needed.
                    // No inverse scaling needed either! We just store the exact 128-bit integer.
                    int c_idx = (i + ii) * ldc + (j + jj);
                    C[c_idx] = exact_int_C_u;
                }
            }
        }
    }

    for (int t = 0; t < NUM_MODULI; t++) {
        free(A_mod[t]); free(B_mod[t]);
    }
}

void reference_u54_gemm(int M, int N, int K, 
                        const uint64_t *A, int lda, 
                        const uint64_t *B, int ldb, 
                        unsigned __int128 *C, int ldc) 
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            unsigned __int128 sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (unsigned __int128)A[i * lda + k] * (unsigned __int128)B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

// Generates a random 54-bit unsigned integer
uint64_t rand_u54() {
    uint64_t r1 = (uint64_t)rand() & 0x7FFF;
    uint64_t r2 = (uint64_t)rand() & 0x7FFF;
    uint64_t r3 = (uint64_t)rand() & 0x7FFF;
    uint64_t r4 = (uint64_t)rand() & 0x7FFF;
    uint64_t val = (r1 << 45) | (r2 << 30) | (r3 << 15) | r4;
    return val & ((1ULL << 54) - 1);
}

// --- Test Driver ---
int main() {
    int M = 128, N = 128, K = 128;
    srand(time(NULL));
    init_crt();

    printf("=== Zvvm Ozaki Scheme II Tests ===\n");
    printf("Matrix Dimensions: %dx%d\n", M, N);
    printf("CRT Constants Initialized.\n\n");

    // ------------------------------------------------------------------------
    // TEST 1: DGEMM (FP64)
    // ------------------------------------------------------------------------
    printf("--- Test 1: FP64 DGEMM ---\n");
    double *A_f = (double*)malloc(M * K * sizeof(double));
    double *B_f = (double*)malloc(K * N * sizeof(double));
    double *C_ref_f = (double*)malloc(M * N * sizeof(double));
    double *C_ozaki_f = (double*)malloc(M * N * sizeof(double));

    for (int i = 0; i < M * K; i++) A_f[i] = (double)(rand() % 2049 - 1024) / 1024.0;
    for (int i = 0; i < K * N; i++) B_f[i] = (double)(rand() % 2049 - 1024) / 1024.0;

    reference_dgemm(M, N, K, 1.0, A_f, K, B_f, N, 0.0, C_ref_f, N);
    ozaki_dgemm(M, N, K, 1.0, A_f, K, B_f, N, 0.0, C_ozaki_f, N);

    double max_rel_err = 0.0, max_c_ref = 0.0;
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_ref_f[i]) > max_c_ref) max_c_ref = fabs(C_ref_f[i]);
    }
    for (int i = 0; i < M * N; i++) {
        double diff = fabs(C_ref_f[i] - C_ozaki_f[i]);
        double rel = diff / fmax(max_c_ref, 1e-16);
        if (rel > max_rel_err) max_rel_err = rel;
    }

    if (max_rel_err < 1e-12) printf("Status: [ PASSED ] - Ozaki FP64 matches Reference DGEMM.\n\n");
    else printf("Status: [ FAILED ] - Large error detected in FP64.\n\n");

    free(A_f); free(B_f); free(C_ref_f); free(C_ozaki_f);

    // ------------------------------------------------------------------------
    // TEST 2: 54-bit Unsigned Integer GEMM
    // ------------------------------------------------------------------------
    printf("--- Test 2: 54-bit Unsigned Integer GEMM ---\n");
    uint64_t *A_u54 = (uint64_t*)malloc(M * K * sizeof(uint64_t));
    uint64_t *B_u54 = (uint64_t*)malloc(K * N * sizeof(uint64_t));
    unsigned __int128 *C_ref_u128 = (unsigned __int128*)malloc(M * N * sizeof(unsigned __int128));
    unsigned __int128 *C_ozaki_u128 = (unsigned __int128*)malloc(M * N * sizeof(unsigned __int128));

    for (int i = 0; i < M * K; i++) A_u54[i] = rand_u54();
    for (int i = 0; i < K * N; i++) B_u54[i] = rand_u54();

    reference_u54_gemm(M, N, K, A_u54, K, B_u54, N, C_ref_u128, N);
    ozaki_u54_gemm(M, N, K, A_u54, K, B_u54, N, C_ozaki_u128, N);

    int mismatches = 0;
    for (int i = 0; i < M * N; i++) {
        if (C_ref_u128[i] != C_ozaki_u128[i]) {
            mismatches++;
        }
    }

    if (mismatches == 0) printf("Status: [ PASSED ] - 0 mismatches. Exact integer match!\n");
    else printf("Status: [ FAILED ] - %d mismatches detected.\n", mismatches);

    free(A_u54); free(B_u54); free(C_ref_u128); free(C_ozaki_u128);

    return 0;
}
