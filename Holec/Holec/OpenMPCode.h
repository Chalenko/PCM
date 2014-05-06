#include "omp.h"

void ParallelOMPTranspose(const double* src, double* dst, const int w, const int h);
void ParallelOMPMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w);
void ParallelOMPBlock1MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs = 500);
void ParallelOMPBlock2MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs = 120);
void ParallelOMPBlock3MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs = 120);
void ParallelOMP1Holec(double* A, int n);
void ParallelOMP1Holec(double* src, double* dst, int n);