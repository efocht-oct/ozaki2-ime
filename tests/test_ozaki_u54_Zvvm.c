#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

// ============================================================================
// --- Hardware Parameters (Zvvm: VLEN=16384, lambda=8, W=4) ---
// ============================================================================
#define M_TILE 64
#define K_EFF 32
#define MAX_N_TILE 64

// ============================================================================
// --- Ozaki Scheme II Parameters ---
// ============================================================================
// 15 coprime moduli are sufficient because their product M is ~2^120.
// The max product of K=128 of 54-bit unsigned ints is ~128 * (2^54)^2 = 2^115.
// Since 2^120 > 2^115, the exact integer will never overflow the CRT reconstruction.
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
    for (int i = 0; i < NUM_MODULI; i++) uCRT_M *= MODULI[i];
    CRT_M = (__int128_t)uCRT_M;

    for (int i = 0; i < NUM_MODULI; i++) {
        unsigned __int128 M_t = uCRT_M / MODULI[i];
        __int128_t rem = (__int128_t)(M_t % MODULI[i]);
        __int128_t y_t = mod_inverse(rem, MODULI[i]);
        CRT_M_t_y_t[i] = (__int128_t)(M_t * y_t);
    }
}

// ============================================================================
// --- Zvvm Matrix Compute Kernel ---
// ============================================================================
static inline void zvvm_kernel_int8_mac(
    int32_t* C_tile, const int8_t* A_mod, const int8_t* B_mod, 
    int K, int lda, int ldb) 
{
#if defined(__riscv)
    // --- REAL RISC-V ZVVM INLINE ASSEMBLY ---
    // vl defines the N dimension for the quad widening instruction.
    // N_TILE = vl / lambda -> 64 = 512 / 8.
    size_t vl = MAX_N_TILE * 8; 

    // 1. Zero out the 64x64 INT32 accumulator (Requires 8 Vector Registers -> LMUL=8)
    // We use LMUL=8 to broadcast zero across v8 through v15.
    asm volatile(
        "vsetvli zero, %0, e32, m8, ta, ma \n"
        "vmv.v.i v8, 0 \n"
        : : "r"(vl) : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
    );

    // 2. Set vtype back to m1 for standard loads and matrix ops
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma \n" : : "r"(vl));

    // 3. Inner loop processing K in chunks of K_EFF = 32
    for (int k = 0; k < K; k += K_EFF) {
        const int8_t* A_ptr = A_mod + k;
        const int8_t* B_ptr = B_mod + k;

        // A tile (64x32) fits in v0 (2048 bytes). B tile (32x64) fits in v4 (2048 bytes).
        asm volatile(
            "vmtl.v v0, (%0), %1 \n"
            "vmttl.v v4, (%2), %3 \n"
            "vqwmmacc.vv v8, v0, v4 \n"
            : 
            : "r"(A_ptr), "r"(lda), "r"(B_ptr), "r"(ldb)
            : "v0", "v4" // Clobbers
        );
    }

    // 4. Store the 64x64 INT32 C tile to memory
    asm volatile(
        "vmts.v v8, (%0), %1 \n" 
        : : "r"(C_tile), "r"(MAX_N_TILE * sizeof(int32_t)) : "memory"
    );

#else
    // --- SOFTWARE MOCK FOR X86/ARM TESTING ---
    memset(C_tile, 0, M_TILE * MAX_N_TILE * sizeof(int32_t));
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
#endif
}

// ============================================================================
// --- Ozaki U54 Zvvm GEMM Implementation ---
// ============================================================================
void ozaki_u54_gemm_zvvm(int M, int N, int K, 
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

    // Map inputs to symmetric INT8 modular representations
    for (int i = 0; i < M * K; i++) {
        for (int t = 0; t < NUM_MODULI; t++) {
            int64_t r = A[i] % MODULI[t];
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
    
    // Hardware Tiled Execution & CRT Reconstruction
    for (int i = 0; i < M; i += M_TILE) {
        for (int j = 0; j < N; j += MAX_N_TILE) {
            
            // Execute the Zvvm instructions for each modulus
            for (int t = 0; t < NUM_MODULI; t++) {
                zvvm_kernel_int8_mac(
                    C_mod[t], &A_mod[t][i * lda], &B_mod[t][j], K, lda, ldb
                );
            }

            // Software CRT Reconstruction
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
                    
                    int c_idx = (i + ii) * ldc + (j + jj);
                    C[c_idx] = exact_int_C_u; // Exactly the 54-bit product sums
                }
            }
        }
    }

    for (int t = 0; t < NUM_MODULI; t++) {
        free(A_mod[t]); free(B_mod[t]);
    }
}

// ============================================================================
// --- Reference Implementation ---
// ============================================================================
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

// ============================================================================
// --- Test Driver ---
// ============================================================================
int main() {
    int M = 128, N = 128, K = 128;
    srand(time(NULL));
    init_crt();

    printf("=== Zvvm Ozaki Scheme II: 54-bit Unsigned GEMM ===\n");
    printf("Matrix Dimensions: %dx%d\n", M, N);

#if defined(__riscv)
    printf("Hardware Context: RISC-V Zvvm Inline Assembly ENABLED\n\n");
#else
    printf("Hardware Context: Host Architecture (x86/ARM) -> Using Software Mock\n\n");
#endif

    uint64_t *A = (uint64_t*)malloc(M * K * sizeof(uint64_t));
    uint64_t *B = (uint64_t*)malloc(K * N * sizeof(uint64_t));
    unsigned __int128 *C_ref = (unsigned __int128*)malloc(M * N * sizeof(unsigned __int128));
    unsigned __int128 *C_ozaki = (unsigned __int128*)malloc(M * N * sizeof(unsigned __int128));

    for (int i = 0; i < M * K; i++) A[i] = rand_u54();
    for (int i = 0; i < K * N; i++) B[i] = rand_u54();

    printf("Computing Reference U54 GEMM...\n");
    reference_u54_gemm(M, N, K, A, K, B, N, C_ref, N);

    printf("Computing Ozaki Scheme II U54 Zvvm GEMM...\n");
    ozaki_u54_gemm_zvvm(M, N, K, A, K, B, N, C_ozaki, N);

    int mismatches = 0;
    for (int i = 0; i < M * N; i++) {
        if (C_ref[i] != C_ozaki[i]) {
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("\nStatus: [ PASSED ] - 0 mismatches. Exact integer match!\n");
    } else {
        printf("\nStatus: [ FAILED ] - %d mismatches detected.\n", mismatches);
    }

    free(A); free(B); free(C_ref); free(C_ozaki);
    return 0;
}
