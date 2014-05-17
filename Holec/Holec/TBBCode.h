#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"
#include "tbb/blocked_range3d.h"
#include "tbb/parallel_reduce.h"

using namespace tbb;

void ParallelTBBTranspose(const double* src, double* dst, const int w, const int h);
double ParallelTBBSum(const double* src, const int len);
double ParallelTBBSclMlt(const double* A, const double* B, const int len);
void ParallelTBBMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w);
void ParallelTBBBlock1MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs = 500);
void ParallelTBBBlock2MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs = 120);
//void ParallelCilkMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w);
//void ParallelCilk2MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w);
//void ParallelCilk3MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w);
//void ParallelCilkBlock1MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs = 500);
//void ParallelCilkBlock2MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs = 120);
//void ParallelCilkHolec(double* A, int n);
//void ParallelCilk1Holec(double* src, double* dst, int n);
//void ParallelCilk2Holec(double* src, double* dst, int n);
//void ParallelCilk3Holec(double* src, double* dst, int n);
//void ParallelCilk4Holec(double* src, double* dst, int n);