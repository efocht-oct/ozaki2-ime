#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <riscv_vector.h>
#include "ozaki_common.h"

// Forward declarations
void ozaki_dgemm(int M, int N, int K, double alpha, 
                 const double *A, int lda, 
                 const double *B, int ldb, 
                 double beta, double *C, int ldc);

void ozaki_sgemm(int M, int N, int K, float alpha, 
                 const float *A, int lda, 
                 const float *B, int ldb, 
                 float beta, float *C, int ldc);

void ozaki_dgemm_l8(int M, int N, int K, double alpha, 
                    const double *A, int lda, 
                    const double *B, int ldb, 
                    double beta, double *C, int ldc);

void ozaki_sgemm_l8(int M, int N, int K, float alpha, 
                    const float *A, int lda, 
                    const float *B, int ldb, 
                    float beta, float *C, int ldc);

// Reference DGEMM
void ref_dgemm(int M, int N, int K, double alpha, 
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
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

// Reference SGEMM
void ref_sgemm(int M, int N, int K, float alpha, 
               const float *A, int lda, 
               const float *B, int ldb, 
               float beta, float *C, int ldc) 
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

// Test runner
int run_test_dgemm(int M, int N, int K, size_t lambda) {
    printf("Testing DGEMM: M=%d, N=%d, K=%d, lambda=%zu\n", M, N, K, lambda);
    
    double *A = (double*)malloc(M * K * sizeof(double));
    double *B = (double*)malloc(K * N * sizeof(double));
    double *C_ozaki = (double*)calloc(M * N, sizeof(double));
    double *C_ref = (double*)calloc(M * N, sizeof(double));

    // Initialize with small integer values to avoid floating-point scaling issues in this simple test
    for (int i = 0; i < M * K; i++) A[i] = (double)(i % 5);
    for (int i = 0; i < K * N; i++) B[i] = (double)(i % 3);

    // Pre-set dynamic lambda in vector unit (simulating setting of lambda at beginning of test in test harness)
    __riscv_vsetlambda(lambda);

    ref_dgemm(M, N, K, 1.0, A, K, B, N, 0.0, C_ref, N);
    
    if (lambda == 8) {
        printf("  Testing ozaki_dgemm_l8...\n");
        ozaki_dgemm_l8(M, N, K, 1.0, A, K, B, N, 0.0, C_ozaki, N);
    } else {
        printf("  Testing ozaki_dgemm (dynamic)...\n");
        ozaki_dgemm(M, N, K, 1.0, A, K, B, N, 0.0, C_ozaki, N);
    }

    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_ozaki[i] - C_ref[i]) > 1e-6) {
            errors++;
            if (errors <= 5) {
                printf("  Mismatch at %d: ref=%f, ozaki=%f\n", i, C_ref[i], C_ozaki[i]);
            }
        }
    }

    free(A); free(B); free(C_ozaki); free(C_ref);
    
    if (errors == 0) {
        printf("  PASSED\n");
        return 0;
    } else {
        printf("  FAILED (%d errors)\n", errors);
        return 1;
    }
}

int run_test_sgemm(int M, int N, int K, size_t lambda) {
    printf("Testing SGEMM: M=%d, N=%d, K=%d, lambda=%zu\n", M, N, K, lambda);
    
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C_ozaki = (float*)calloc(M * N, sizeof(float));
    float *C_ref = (float*)calloc(M * N, sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 5);
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 3);

    // Pre-set dynamic lambda in vector unit (simulating setting of lambda at beginning of test in test harness)
    __riscv_vsetlambda(lambda);

    ref_sgemm(M, N, K, 1.0f, A, K, B, N, 0.0f, C_ref, N);

    if (lambda == 8) {
        printf("  Testing ozaki_sgemm_l8...\n");
        ozaki_sgemm_l8(M, N, K, 1.0f, A, K, B, N, 0.0f, C_ozaki, N);
    } else {
        printf("  Testing ozaki_sgemm (dynamic)...\n");
        ozaki_sgemm(M, N, K, 1.0f, A, K, B, N, 0.0f, C_ozaki, N);
    }

    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (fabsf(C_ozaki[i] - C_ref[i]) > 1e-4f) {
            errors++;
            if (errors <= 5) {
                printf("  Mismatch at %d: ref=%f, ozaki=%f\n", i, C_ref[i], C_ozaki[i]);
            }
        }
    }

    free(A); free(B); free(C_ozaki); free(C_ref);
    
    if (errors == 0) {
        printf("  PASSED\n");
        return 0;
    } else {
        printf("  FAILED (%d errors)\n", errors);
        return 1;
    }
}

int main(int argc, char *argv[]) {
    // Disable buffering on stdout to ensure prints show up immediately
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    // Default values
    size_t lambda = 8;

    // Parse command line arguments
    // Usage: test_ozaki [lambda] [VLEN]
    if (argc > 1) {
        lambda = (size_t)atoi(argv[1]);
    }
    if (argc > 2) {
        // VLEN argument is ignored - detected from hardware
    }

    const char *env_lambda = getenv("LAMBDA");
    if (!env_lambda && argc <= 1) {
        lambda = 8;
    } else if (env_lambda && argc <= 1) {
        lambda = (size_t)atoi(env_lambda);
    }

    printf("Ozaki-2 IME Test Suite\n");
    printf("======================\n");
    
    // Detect hardware VLEN at runtime
    size_t vlmax_e32m1 = __riscv_vsetvl_e32m1(~0ULL);
    size_t hw_vlen = vlmax_e32m1 * 32;
    printf("Hardware VLEN=%zu, Configured lambda=%zu\n", hw_vlen, lambda);

    int total_errors = 0;

    // Test the specific lambda value (passed from outside)
    printf("\n--- Testing with lambda=%zu ---\n", lambda);
    
    // Small matrix tests
    total_errors += run_test_dgemm(64, 64, 64, lambda);
    total_errors += run_test_sgemm(64, 64, 64, lambda);
    
    // Larger matrix tests (up to 16384 VLEN support)
    // Note: Full 16384 VLEN testing requires large memory, using 128x128 for CI speed
    total_errors += run_test_dgemm(128, 128, 128, lambda);
    total_errors += run_test_sgemm(128, 128, 128, lambda);

    printf("\n======================\n");
    if (total_errors == 0) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("TESTS FAILED: %d errors\n", total_errors);
        return 1;
    }
}
