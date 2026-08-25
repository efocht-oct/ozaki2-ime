#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <riscv_vector.h>
#include "ozaki_common.h"

/* E4M3 is used as an exact transport format for signed digits in [-16,16].
 * A residue r is represented as r = lo + 16*hi, with lo=r%16. */
static uint8_t fp8_e4m3_int(int x) {
    if (x == 0) return 0;
    int sign = x < 0; if (sign) x = -x;
    int e = 0; while ((1 << (e + 1)) <= x) ++e;
    int mant = (x - (1 << e)) * 8 / (1 << e);
    return (uint8_t)((sign << 7) | ((e + 7) << 3) | mant);
}
static void split_residue(int r, uint8_t *lo, uint8_t *hi) {
    int l = r % 16;
    *lo = fp8_e4m3_int(l);
    *hi = fp8_e4m3_int((r - l) / 16);
}

#ifdef MOCK_IME
static void fp8_kernel(float *out, const uint8_t *a, const uint8_t *b,
                       int K, int lda, int ldb, size_t M, size_t N) {
    for (size_t i=0; i<M; ++i) for (size_t j=0; j<N; ++j) {
        float s=0;
        for (int k=0;k<K;++k) {
            /* Mock arrays contain E4M3 encodings of small integers. */
            int8_t av=(int8_t)a[i*(size_t)lda+k], bv=(int8_t)b[j*(size_t)ldb+k];
            (void)av; (void)bv;
            /* The caller supplies decoded digit arrays in MOCK_IME. */
            s += ((const int8_t *)a)[i*(size_t)lda+k] * ((const int8_t *)b)[j*(size_t)ldb+k];
        }
        out[i*N+j]=s;
    }
}
#else
#define KERNEL(CS, CT, VL, VMV, MAC, STORE) \
static void fp8_kernel_##CS(float *out,const uint8_t *a,const uint8_t *b,int K, \
 int lda,int ldb,size_t M,size_t N,size_t lambda,size_t keff,size_t astride,size_t bstride){ \
 size_t vl=VL(M*N); CT acc=VMV(0.0,vl); \
 for(size_t k0=0,p=0;k0<(size_t)K;k0+=keff,++p){ \
  size_t va_vl=__riscv_vsetvl_e8m1(M*keff); __riscv_vsetlambda(lambda); \
  vfloat8e4m3m1_t va=__riscv_vmtl_v_f8e4m3m1(a+p*astride,lda,va_vl); \
  size_t vb_vl=__riscv_vsetvl_e8m1(N*keff); __riscv_vsetlambda(lambda); \
  vfloat8e4m3m1_t vb=__riscv_vmtl_v_f8e4m3m1(b+p*bstride,ldb,vb_vl); \
  acc=MAC(acc,va,vb,vl); } STORE(out,N,acc,vl); }
KERNEL(m1,vfloat32m1_t,__riscv_vsetvl_e32m1,__riscv_vfmv_v_f_f32m1,__riscv_vfqwmmacc_vv_f32m1_f8e4m3m1,__riscv_vmts_v_f32m1)
KERNEL(m2,vfloat32m2_t,__riscv_vsetvl_e32m2,__riscv_vfmv_v_f_f32m2,__riscv_vfqwmmacc_vv_f32m2_f8e4m3m1,__riscv_vmts_v_f32m2)
KERNEL(m4,vfloat32m4_t,__riscv_vsetvl_e32m4,__riscv_vfmv_v_f_f32m4,__riscv_vfqwmmacc_vv_f32m4_f8e4m3m1,__riscv_vmts_v_f32m4)
KERNEL(m8,vfloat32m8_t,__riscv_vsetvl_e32m8,__riscv_vfmv_v_f_f32m8,__riscv_vfqwmmacc_vv_f32m8_f8e4m3m1,__riscv_vmts_v_f32m8)
#undef KERNEL
static void fp8_kernel(float *out,const uint8_t*a,const uint8_t*b,int K,int lda,int ldb,size_t M,size_t N,size_t lambda,size_t keff,size_t as,size_t bs){
 size_t emul=__riscv_vsetvl_e32m1(~0ULL)/(lambda*lambda);
 switch(emul){case 1:fp8_kernel_m1(out,a,b,K,lda,ldb,M,N,lambda,keff,as,bs);break;case 2:fp8_kernel_m2(out,a,b,K,lda,ldb,M,N,lambda,keff,as,bs);break;case 4:fp8_kernel_m4(out,a,b,K,lda,ldb,M,N,lambda,keff,as,bs);break;default:fp8_kernel_m8(out,a,b,K,lda,ldb,M,N,lambda,keff,as,bs);break;}}
#endif

void ozaki_dgemm(int M,int N,int K,double alpha,const double*A,int lda,const double*B,int ldb,double beta,double*C,int ldc){
 size_t lambda=__riscv_vsetlambda(0); if(!lambda) lambda=8; size_t vlmax=__riscv_vsetvl_e32m1(~0ULL); size_t mt=vlmax/lambda, nt=mt, W=4, keff=lambda*W, lines=lambda;
 if(!mt)mt=nt=1; size_t panels=((size_t)K+keff-1)/keff, astride=W*(size_t)M*lines, bstride=W*(size_t)N*lines;
 uint8_t *am[FP64_NUM_MODULI][2],*bm[FP64_NUM_MODULI][2]; float *cm[FP64_NUM_MODULI];
 for(int t=0;t<FP64_NUM_MODULI;++t)for(int q=0;q<2;++q){
#ifdef MOCK_IME
  am[t][q]=malloc((size_t)M*K); bm[t][q]=malloc((size_t)K*N);
#else
  am[t][q]=calloc(panels*astride,1); bm[t][q]=calloc(panels*bstride,1);
#endif
 } for(int t=0;t<FP64_NUM_MODULI;++t)cm[t]=malloc(mt*nt*sizeof(float));
 double ma=0,mb=0; for(int i=0;i<M;++i)for(int k=0;k<K;++k)ma=fmax(ma,fabs(A[i*lda+k])); for(int k=0;k<K;++k)for(int j=0;j<N;++j)mb=fmax(mb,fabs(B[k*ldb+j]));
 double sa=ma?exp2(-floor(log2(ma))+52):1, sb=mb?exp2(-floor(log2(mb))+52):1;
 for(int i=0;i<M;++i)for(int k=0;k<K;++k){int64_t v=llround(A[i*lda+k]*sa);for(int t=0;t<FP64_NUM_MODULI;++t){int r=v%MODULI_FP64[t];if(r<0)r+=MODULI_FP64[t];if(r>MODULI_FP64[t]/2)r-=MODULI_FP64[t];uint8_t lo,hi;split_residue(r,&lo,&hi);
#ifdef MOCK_IME
 am[t][0][i*K+k]=(uint8_t)(int8_t)(r%16); am[t][1][i*K+k]=(uint8_t)(int8_t)((r-(r%16))/16);
#else
 size_t p=k/keff,z=k%keff,tr=W*(size_t)i+z/lines,tc=z%lines;am[t][0][p*astride+tr*lines+tc]=lo;am[t][1][p*astride+tr*lines+tc]=hi;
#endif
 }}
 for(int k=0;k<K;++k)for(int j=0;j<N;++j){int64_t v=llround(B[k*ldb+j]*sb);for(int t=0;t<FP64_NUM_MODULI;++t){int r=v%MODULI_FP64[t];if(r<0)r+=MODULI_FP64[t];if(r>MODULI_FP64[t]/2)r-=MODULI_FP64[t];uint8_t lo,hi;split_residue(r,&lo,&hi);
#ifdef MOCK_IME
 bm[t][0][j*K+k]=(uint8_t)(int8_t)(r%16); bm[t][1][j*K+k]=(uint8_t)(int8_t)((r-(r%16))/16);
#else
 size_t p=k/keff,z=k%keff,tr=W*(size_t)j+z/lines,tc=z%lines;bm[t][0][p*bstride+tr*lines+tc]=lo;bm[t][1][p*bstride+tr*lines+tc]=hi;
#endif
 }}
 for(int i=0;i<M;i+=mt)for(int j=0;j<N;j+=nt){size_t mm=(M-i<mt)?M-i:mt,nn=(N-j<nt)?N-j:nt;for(int t=0;t<FP64_NUM_MODULI;++t){float s00[mt*nt],s01[mt*nt],s10[mt*nt],s11[mt*nt];
#ifdef MOCK_IME
 fp8_kernel(s00,(uint8_t*)&am[t][0][i*K],(uint8_t*)&bm[t][0][j*K],K,K,K,mm,nn);fp8_kernel(s01,(uint8_t*)&am[t][0][i*K],(uint8_t*)&bm[t][1][j*K],K,K,K,mm,nn);fp8_kernel(s10,(uint8_t*)&am[t][1][i*K],(uint8_t*)&bm[t][0][j*K],K,K,K,mm,nn);fp8_kernel(s11,(uint8_t*)&am[t][1][i*K],(uint8_t*)&bm[t][1][j*K],K,K,K,mm,nn);
#else
 fp8_kernel(s00,&am[t][0][i*W*lines],&bm[t][0][j*W*lines],K,lines,lines,mm,nn,lambda,keff,astride,bstride);fp8_kernel(s01,&am[t][0][i*W*lines],&bm[t][1][j*W*lines],K,lines,lines,mm,nn,lambda,keff,astride,bstride);fp8_kernel(s10,&am[t][1][i*W*lines],&bm[t][0][j*W*lines],K,lines,lines,mm,nn,lambda,keff,astride,bstride);fp8_kernel(s11,&am[t][1][i*W*lines],&bm[t][1][j*W*lines],K,lines,lines,mm,nn,lambda,keff,astride,bstride);
#endif
 for(size_t ii=0;ii<mm;++ii)for(size_t jj=0;jj<nn;++jj){cm[t][ii*nt+jj]=s00[ii*nn+jj]+16*(s01[ii*nn+jj]+s10[ii*nn+jj])+256*s11[ii*nn+jj];}}
 for(size_t ii=0;ii<mm;++ii)for(size_t jj=0;jj<nn;++jj){unsigned __int128 Mcrt=CRT_M_FP64,u=0;for(int t=0;t<FP64_NUM_MODULI;++t){int32_t v=(int32_t)llround(cm[t][ii*nt+jj])%MODULI_FP64[t];if(v<0)v+=MODULI_FP64[t];u=(u+mult_mod((unsigned __int128)CRT_M_T_Y_T_FP64[t],v,Mcrt))%Mcrt;}__int128 z=(__int128)u;if(z>(__int128)(Mcrt/2))z-=(__int128)Mcrt;double x=(double)z/(sa*sb);size_t idx=(i+ii)*ldc+j+jj;C[idx]=alpha*x+beta*C[idx];}}
 for(int t=0;t<FP64_NUM_MODULI;++t){for(int q=0;q<2;++q){free(am[t][q]);free(bm[t][q]);}free(cm[t]);}}
void ozaki_dgemm_l8(int M,int N,int K,double a,const double*A,int lda,const double*B,int ldb,double b,double*C,int ldc){ozaki_dgemm(M,N,K,a,A,lda,B,ldb,b,C,ldc);}
