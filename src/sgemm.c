#include <stddef.h>
#include <stdlib.h>
#include <riscv_vector.h>

void ozaki_dgemm(int, int, int, double, const double *, int, const double *, int,
                 double, double *, int);

/* SGEMM uses the same E4M3-digit FP8 engine as DGEMM.  The shared kernel uses
 * FP32 accumulator tiles, matching the only floating-point accumulator
 * supported by the target architecture. */
void ozaki_sgemm(int M, int N, int K, float alpha, const float *A, int lda,
                 const float *B, int ldb, float beta, float *C, int ldc) {
    double *ad = malloc((size_t)M * K * sizeof(*ad));
    double *bd = malloc((size_t)K * N * sizeof(*bd));
    double *cd = malloc((size_t)M * N * sizeof(*cd));
    if (!ad || !bd || !cd) { free(ad); free(bd); free(cd); return; }
    for (int i=0;i<M;++i) for (int k=0;k<K;++k) ad[(size_t)i*K+k]=A[(size_t)i*lda+k];
    for (int k=0;k<K;++k) for (int j=0;j<N;++j) bd[(size_t)k*N+j]=B[(size_t)k*ldb+j];
    for (int i=0;i<M;++i) for (int j=0;j<N;++j) cd[(size_t)i*N+j]=C[(size_t)i*ldc+j];
    ozaki_dgemm(M,N,K,(double)alpha,ad,K,bd,N,(double)beta,cd,N);
    for (int i=0;i<M;++i) for (int j=0;j<N;++j) C[(size_t)i*ldc+j]=(float)cd[(size_t)i*N+j];
    free(ad); free(bd); free(cd);
}
void ozaki_sgemm_l8(int M,int N,int K,float a,const float*A,int lda,const float*B,int ldb,float b,float*C,int ldc){
    ozaki_sgemm(M,N,K,a,A,lda,B,ldb,b,C,ldc);
}
