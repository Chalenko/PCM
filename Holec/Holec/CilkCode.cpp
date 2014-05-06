#include "stdafx.h"
#include "CilkCode.h"`

void ParallelCilkTranspose(const double* src, double* dst, const int w, const int h){
	for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++){
            dst[i * w + j] = src[j * h + i];
		}
	}
}

void ParallelCilkMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w){
	double* tr = new double [src1w * src2w];
	ParallelCilkTranspose(src2, tr, src2w, src1w);
	int i, j, k;
	for (i = 0; i < src1h; i++){
		for (j = 0; j < src2w; j++){
			double sum = 0;
			for (k = 0; k < src1w; k++){
				sum += src1[i*src1w + k] * tr[j*src1w + k];
			}
			dst[i*src2w + j] = sum;
		}
	}
	delete [] tr;
}

void ParallelCilkHolec(double* A, int n){
	for(int i = 0; i < n; i++){
		double sum = 0;
		for (int j = 0; j < i; j++){
			sum = 0;
			for (int k = 0; k < j; k++){
				sum += A[i*n+k]*A[j*n+k];
			}
			A[i*n+j] = (A[i*n+j]-sum)/A[j*n+j];
		}
		sum = 0;
		for (int k = 0; k < i; k++){
			sum += A[i*n+k]*A[i*n+k];
		}
		A[i*n+i] = sqrt(A[i*n+i] - sum);
	}
}

void ParallelCilk1Holec(double* src, double* dst, int n){
	#pragma cilk grainsize = 32
	cilk_for(int i = 0; i < n; i++){
		double sum = 0;
		for (int j = 0; j < i; j++){
			sum = 0;
			for (int k = 0; k < j; k++){
				sum += dst[i*n+k]*dst[j*n+k];
			}
			dst[i*n+j] = (src[i*n+j]-sum)/dst[j*n+j];
			dst[j*n+i] = 0;
		}
		sum = 0;
		for (int k = 0; k < i; k++){
			sum += dst[i*n+k]*dst[i*n+k];
		}
		dst[i*n+i] = sqrt(src[i*n+i] - sum);
	}
}

void ParallelCilk2Holec(double* src, double* dst, int n){
	cilk_for(int i = 0; i < n; i++){
		double sum = 0;
		for (int j = 0; j < i; j++){
			//sum = __sec_reduce_add(dst[i*n:j]*dst[j*n:j]);
			dst[i*n+j] = (src[i*n+j]-sum)/dst[j*n+j];
			//dst[j*n+i] = 0;
		}
		//sum = __sec_reduce_add(dst[i*n:i]*dst[i*n:i]);
		dst[i*n+i] = sqrt(src[i*n+i] - sum);
		for (int j = i+1; j < n; j++){
			dst[i*n+j] = 0;
		}
	}
}

void ParallelCilk3Holec(double* src, double* dst, int n){
	cilk_for(int i = 0; i < n; i++){
		double sum = 0;
		#pragma simd
		for (int j = 0; j < i; j++){
			//#pragma simd
			//sum = __sec_reduce_add(dst[i*n:j]*dst[j*n:j]);
			dst[i*n+j] = (src[i*n+j]-sum)/dst[j*n+j];
			//dst[j*n+i] = 0;
		}
		//sum = __sec_reduce_add(dst[i*n:i]*dst[i*n:i]);
		dst[i*n+i] = sqrt(src[i*n+i] - sum);
		//dst[(n*i+i+1):(n-i-1):1] = 0;
	}
}

void ParallelCilk4Holec(double* src, double* dst, int n){
	cilk_for(int i = 0; i < n; i++){
		cilk::reducer_opadd <double> sum(0);
		for (int j = 0; j < i; j++){
			sum.set_value(0);// = 0;
			for (int k = 0; k < j; k++){
				sum += dst[i*n+k]*dst[j*n+k];
			}
			dst[i*n+j] = (src[i*n+j]-sum.get_value())/dst[j*n+j];
			dst[j*n+i] = 0;
		}
		sum.set_value(0);// = 0;
		for (int k = 0; k < i; k++){
			sum += dst[i*n+k]*dst[i*n+k];
		}
		dst[i*n+i] = sqrt(src[i*n+i] - sum.get_value());
	}
}