#include <cilk\cilk.h>
#include <cilk\reducer_opadd.h>

void ParallelCilkTranspose(const double* src, double* dst, const int w, const int h);
void ParallelCilkMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w);
void ParallelCilk2MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w);
void ParallelCilk3MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w);
void ParallelCilkBlock1MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs = 500);
void ParallelCilkBlock2MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs = 120);
void ParallelCilkHolec(double* A, int n);
void ParallelCilk1Holec(double* src, double* dst, int n);
void ParallelCilk2Holec(double* src, double* dst, int n);
void ParallelCilk3Holec(double* src, double* dst, int n);
void ParallelCilk4Holec(double* src, double* dst, int n);