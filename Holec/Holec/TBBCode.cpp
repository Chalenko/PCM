#include "stdafx.h"
#include "TBBCode.h"

class VectorAdditioner{ 
	const double *vector1, *vector2;
	double *const result;
	int const len;
	
	public: 
		VectorAdditioner(const double *tvector1, const double *tvector2, double *tresult, int tlen) : vector1(tvector1), vector2(tvector2), result(tresult), len(tlen){
		}
		void operator()(const blocked_range<int>& r) const{
			int begin = r.begin(), end = r.end();
			for (int i = begin; i < end; i++){
				result[i] = vector1[i] + vector2[i];
			}
		}
};

void ParallelTBBAdd(const double *A, const double *B, double *dst, const int w, const int h)
{
	int len = h * w;
	parallel_for(blocked_range<int>(0, len, 40), VectorAdditioner(A, B, dst, len), affinity_partitioner());
}

class VectorAdditionerEq{ 
	const double *vector;
	double *const result;
	int const len;
	
	public: 
		VectorAdditionerEq(const double *tvector, double *tresult, int tlen) : vector(tvector), result(tresult), len(tlen){
		}
		void operator()(const blocked_range<int>& r) const{
			int begin = r.begin(), end = r.end();
			for (int i = begin; i < end; i++){
				result[i] += vector[i] ;
			}
		}
};

void ParallelTBBAddEq(const double *A, double *dst, const int w, const int h)
{
	int len = h * w;
	parallel_for(blocked_range<int>(0, len, 40), VectorAdditionerEq(A, dst, len), affinity_partitioner());
}

class MatrixTransposer1{ 
	const double *src;
	double *const dst;
	int const wN;
	int const hN;
	
	public: 
		MatrixTransposer1(const double *tsrc, double *tdst, int twN, int thN) : src(tsrc), dst(tdst), wN(twN), hN(thN){
		}
		void operator()(const blocked_range2d<int>& r) const{
			int begin1 = r.rows().begin(), end1 = r.rows().end();
			int begin2 = r.cols().begin(), end2 = r.cols().end();
			for (int i = begin1; i < end1; i++){
				for (int j = begin2; j < end2; j++){
					dst[i * wN + j] = src[j * hN + i];
				}
			}
		}
};

void TransposeCol(const double *src, double *dstvec, int col, int wN, int hN){
	for (int j = 0; j < wN; j++){
		dstvec[j] = src[j * hN + col];
	}
}

class MatrixTransposer2{ 
	const double *src;
	double *const dst;
	int const wN;
	int const hN;
	
	public: 
		MatrixTransposer2(const double *tsrc, double *tdst, int twN, int thN) : src(tsrc), dst(tdst), wN(twN), hN(thN){
		}
		void operator()(const blocked_range<int>& r) const{
			int begin = r.begin(), end = r.end();
			for (int i = begin; i < end; i++){
				TransposeCol(src, &dst[i * wN], i, wN, hN);
			}
		}
};

void ParallelTBBTranspose(const double* src, double* dst, const int wN, const int hN){
	//parallel_for(blocked_range2d<int>(0, hN, 0, wN), MatrixTransposer1(src, dst, wN, hN), affinity_partitioner());
	parallel_for(blocked_range<int>(0, hN, 40), MatrixTransposer2(src, dst, wN, hN), affinity_partitioner());
}

class VectorSummator{
	private:
		const double *a;
		double c;

	public:
		explicit VectorSummator(const double *ta) : a(ta), c(0){
		}

		VectorSummator(const VectorSummator& vs, split) : a(vs.a), c(0){
		}

		void operator()(const blocked_range<int>& r){
			int begin = r.begin(), end = r.end();
			for (int i = begin; i < end; i++){
				c += a[i];
			}
		}

		void join(const VectorSummator& sum){
			c += sum.c;
		}

		double Result(){
			return c;
		}
};

double ParallelTBBSum(const double* src, const int len){
	VectorSummator sum(src);
	parallel_reduce(blocked_range<int>(0, len), sum);
	return sum.Result();
}

double TBBSclMlt(const double* A, const double* B, const int len){
	double result = 0;
	for(int k = 0; k < len; k++){
		result += A[k] * B[k];
	}
	return result;
}

class ScalarMultiplicator{
	private:
		const double *a, *b;
		double c;

	public:
		explicit ScalarMultiplicator(const double *ta, const double *tb) : a(ta), b(tb), c(0){
		}

		ScalarMultiplicator(const ScalarMultiplicator& sm, split) : a(sm.a), b(sm.b), c(0){
		}

		void operator()(const blocked_range<int>& r){
			int begin = r.begin();
			int end = r.end();
			c += TBBSclMlt(&(a[begin]), &(b[begin]), (end - begin));
		}

		void join(const ScalarMultiplicator& mul){
			c += mul.c;
		}

		double Result(){
			return c;
		}
};

double ParallelTBBSclMlt(const double* A, const double* B, const int len){
	ScalarMultiplicator mul(A, B);
	parallel_reduce(blocked_range<int>(0, len), mul);
	return mul.Result();
}

class MatrixMultiplicator1{ 
	const double *A, *B;
	double *const dst;
	int const src1h, src1w, src2w;
	
	public: 
		MatrixMultiplicator1(const double *tA, const double *tB, double *tdst, const int tsrc1h, const int tsrc1w, const int tsrc2w) : A(tA), B(tB), dst(tdst), src1h(tsrc1h), src1w(tsrc1w), src2w(tsrc2w){
		}
		void operator()(const blocked_range2d<int>& r) const{
			int begin1 = r.rows().begin(), end1 = r.rows().end();
			int begin2 = r.cols().begin(), end2 = r.cols().end();
			for (int i = begin1; i < end1; i++){
				for (int j = begin2; j < end2; j++){
					const double *vec1, *vec2;
					vec1 = &(A[i*src1w]);
					vec2 = &(B[j*src1w]);
					dst[i * src2w + j] = TBBSclMlt(vec1, vec2, src1w);
				}
			}
		}
};

class MatrixMultiplicator2{ 
	const double *A, *B;
	double *const dst;
	int const src1h, src1w, src2w;
	
	public: 
		MatrixMultiplicator2(const double *tA, const double *tB, double *tdst, const int tsrc1h, const int tsrc1w, const int tsrc2w) : A(tA), B(tB), dst(tdst), src1h(tsrc1h), src1w(tsrc1w), src2w(tsrc2w){
		}
		void operator()(const blocked_range<int>& r) const{
			int begin = r.begin(), end = r.end();
			for (int i = begin; i < end; i++){
				for (int j = 0; j < src2w; j++){
					const double *vec1, *vec2;
					vec1 = &(A[i*src1w]);
					vec2 = &(B[j*src1w]);
					dst[i * src2w + j] = TBBSclMlt(vec1, vec2, src1w);
				}
			}
		}
};

void ParallelTBBMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w){
	double* tr = new double [src1w * src2w];
	ParallelTBBTranspose(src2, tr, src2w, src1w);
	//parallel_for(blocked_range2d<int>(0, src1h, 0, src2w), MatrixMultiplicator1(src1, tr, dst, src1h, src1w, src2w));
	parallel_for(blocked_range<int>(0, src1h), MatrixMultiplicator2(src1, tr, dst, src1h, src1w, src2w));
	delete [] tr;
}

class Equaler{ 
	double *const dst;
	int const value;
	
	public: 
		Equaler(double *tdst, const int tvalue) : dst(tdst), value(tvalue){
		}
		void operator()(const blocked_range<int>& r) const{
			int begin = r.begin(), end = r.end();
			for (int i = begin; i < end; i++){
				dst[i] = value;
			}
		}
};

class BlockMatrixMultiplicator{ 
	const double *A, *B;
	double *const dst;
	int const src1h, src1w, src2w;
	
	public: 
		BlockMatrixMultiplicator(const double *tA, const double *tB, double *tdst, const int tsrc1h, const int tsrc1w, const int tsrc2w) : A(tA), B(tB), dst(tdst), src1h(tsrc1h), src1w(tsrc1w), src2w(tsrc2w){
		}
		void operator()(const blocked_range3d<int>& r) const{
			int begin1 = r.rows().begin(), end1 = r.rows().end();
			int begin2 = r.cols().begin(), end2 = r.cols().end();
			int begin3 = r.pages().begin(), end3 = r.pages().end();
			for (int i = begin1; i < end1; i++){
				for (int j = begin2; j < end2; j++){
					/*
					for(int k = begin3; k < end3; k++){
						dst[i * src2w + j] += A[k] * B[k];
					}
					*/
					/*
					const double *vec1, *vec2;
					vec1 = &(A[i * src1w]);
					vec2 = &(B[j * src1w]);
					ScalarMultiplicator mul(vec1, vec2);
					parallel_reduce(r.pages(), mul);
					dst[i * src2w + j] = mul.Result();
					*/
					/*
					const double *vec1, *vec2;
					vec1 = &(A[i * src1w]);
					vec2 = &(B[j * src1w]);
					dst[i * src2w + j] = TBBSclMlt(vec1, vec2, src1w);
					*/
					const double *vec1, *vec2;
					vec1 = &(A[i * src1w + begin3]);
					vec2 = &(B[j * src1w + begin3]);
					dst[i * src2w + j] += TBBSclMlt(vec1, vec2, end3 - begin3);
				}
			}
		}
};

void ParallelTBBBlock2MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs){
	int blockSize = bs;

	double *tB = new double[src2w * src1w];
	ParallelTBBTranspose(src2, tB, src1w, src2w);

	parallel_for(blocked_range<int>(0, (src1h * src2w)), Equaler(dst, 0));

	parallel_for(blocked_range3d<int>(0, src1w, bs, 0, src1h, bs, 0, src2w, bs), BlockMatrixMultiplicator(src1, tB, dst, src1h, src1w, src2w));

	delete[] tB;
}