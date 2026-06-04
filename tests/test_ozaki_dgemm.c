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

// --- Ozaki Scheme II DGEMM ---
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

    // 1. Exponent Scaling
    double max_A = 0.0, max_B = 0.0;
    for (int i = 0; i < M * K; i++) max_A = fmax(max_A, fabs(A[i]));
    for (int i = 0; i < K * N; i++) max_B = fmax(max_B, fabs(B[i]));
    
    // Scale factors to map FP64 mantissas to integers
    // Protect against exact zero matrices to prevent log2(0) = -inf
    double scale_A = (max_A == 0.0) ? 1.0 : exp2(-floor(log2(max_A)) + 52); 
    double scale_B = (max_B == 0.0) ? 1.0 : exp2(-floor(log2(max_B)) + 52);

    // 2. Modular Reduction
    for (int i = 0; i < M * K; i++) {
        int64_t a_int = (int64_t)round(A[i] * scale_A);
        for (int t = 0; t < NUM_MODULI; t++) {
            int64_t r = a_int % MODULI[t];
            if (r < 0) r += MODULI[t];                // Standard positive modulo
            if (r > MODULI[t] / 2) r -= MODULI[t];   // Symmetric modulo for valid INT8 mapping
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

    // Workspace for 64x64 C block mod m_t
    int32_t C_mod[NUM_MODULI][M_TILE * MAX_N_TILE];
    
    // 3. Tiled Matrix Multiplication & 4. CRT Reconstruction
    for (int i = 0; i < M; i += M_TILE) {
        for (int j = 0; j < N; j += MAX_N_TILE) {
            
            // Zvvm Mock Compute
            for (int t = 0; t < NUM_MODULI; t++) {
                zvvm_kernel_int8_mac(
                    C_mod[t], 
                    &A_mod[t][i * lda], 
                    &B_mod[t][j],
                    K, lda, ldb
                );
            }

            // CRT Reconstruction
            for (int ii = 0; ii < M_TILE && (i + ii) < M; ii++) {
                for (int jj = 0; jj < MAX_N_TILE && (j + jj) < N; jj++) {
                    unsigned __int128 exact_int_C_u = 0;
                    unsigned __int128 uCRT_M = (unsigned __int128)CRT_M;

                    for (int t = 0; t < NUM_MODULI; t++) {
                        int64_t val = C_mod[t][ii * MAX_N_TILE + jj];
                        val %= MODULI[t];
                        if (val < 0) val += MODULI[t]; // Ensure positive remainder
                        
                        // Accumulate using unsigned 128-bit integer to prevent overflow
                        unsigned __int128 term = (unsigned __int128)val * (unsigned __int128)CRT_M_t_y_t[t];
                        exact_int_C_u = (exact_int_C_u + (term % uCRT_M)) % uCRT_M;
                    }
                    
                    __int128_t exact_int_C = (__int128_t)exact_int_C_u;
                    if (exact_int_C > CRT_M / 2) {
                        exact_int_C -= CRT_M;
                    }

                    // 5. Inverse Scaling
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

    for (int t = 0; t < NUM_MODULI; t++) {
        free(A_mod[t]);
        free(B_mod[t]);
    }
}

// --- Standard Reference DGEMM ---
void reference_dgemm(int M, int N, int K, double alpha, 
                     const double *A, int lda, 
                     const double *B, int ldb, 
                     double beta, double *C, int ldc) 
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            if (beta == 0.0) {
                C[i * ldc + j] = alpha * sum;
            } else {
                C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
            }
        }
    }
}

// --- Test Driver ---
int main() {
    int M = 128, N = 128, K = 128;
    double alpha = 1.0, beta = 0.0;

    printf("=== Zvvm Ozaki Scheme II DGEMM Test ===\n");
    printf("Matrix Dimensions: %dx%d\n", M, N);
    
    init_crt();
    printf("CRT Constants Initialized.\n");

    // Allocate matrices
    double *A = (double*)malloc(M * K * sizeof(double));
    double *B = (double*)malloc(K * N * sizeof(double));
    double *C_ref = (double*)malloc(M * N * sizeof(double));
    double *C_ozaki = (double*)malloc(M * N * sizeof(double));

    // Initialize with exact binary fractions (multiples of 1/1024)
    // This perfectly prevents floating-point accumulation rounding errors in 
    // the reference DGEMM, allowing a true 0.0 bitwise match comparison.
    srand(time(NULL));
    for (int i = 0; i < M * K; i++) {
        A[i] = (double)(rand() % 2049 - 1024) / 1024.0;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (double)(rand() % 2049 - 1024) / 1024.0;
    }

    printf("Computing Reference DGEMM...\n");
    reference_dgemm(M, N, K, alpha, A, K, B, N, beta, C_ref, N);

    printf("Computing Ozaki Scheme II DGEMM...\n");
    ozaki_dgemm(M, N, K, alpha, A, K, B, N, beta, C_ozaki, N);

    // Verify Results
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    double max_c_ref = 0.0;

    // Find max value in reference matrix to compute a stable global relative error
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_ref[i]) > max_c_ref) max_c_ref = fabs(C_ref[i]);
    }

    for (int i = 0; i < M * N; i++) {
        double diff = fabs(C_ref[i] - C_ozaki[i]);
        if (diff > max_abs_err) max_abs_err = diff;
        
        // Evaluate relative error globally to prevent spikes from divide-by-zero
        double rel = diff / fmax(max_c_ref, 1e-16);
        if (rel > max_rel_err) max_rel_err = rel;
    }

    printf("\n=== Results ===\n");
    printf("Max Absolute Error: %e\n", max_abs_err);
    printf("Max Relative Error: %e\n", max_rel_err);

    if (max_rel_err < 1e-12) {
        printf("\nStatus: [ PASSED ] - Ozaki Scheme matches Reference DGEMM.\n");
    } else {
        printf("\nStatus: [ FAILED ] - Large error detected.\n");
    }

    free(A); free(B); free(C_ref); free(C_ozaki);
    return 0;
}
