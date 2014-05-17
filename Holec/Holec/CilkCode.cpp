#include "stdafx.h"
#include "CilkCode.h"

void ParallelCilkAdd(const double *A, const double *B, double *dst, const int w, const int h)
{
	dst[0:(w * h)] = A[0:(w * h)] + B[0:(w * h)];
}

void ParallelCilkAddEq(const double *A, double *dst, const int w, const int h)
{
	dst[0:(w * h)] += A[0:(w * h)];
}

void ParallelCilkTranspose(const double* src, double* dst, const int wN, const int hN){
	cilk_for (int i = 0; i < hN; i++){
        cilk_for (int j = 0; j < wN; j++){
            dst[i * wN + j] = src[j * hN + i];
		}
	}
}

double ParallelCilkSum(const double* src, const int len){
	cilk::reducer_opadd<double> sum(0);
	cilk_for (int i = 0; i < len; i++){
		sum += src[i];
	}
	return sum.get_value();
}

double ParallelCilkSclMlt(const double* A, const double* B, const int len){
	return __sec_reduce_add(A[0:len] * B[0:len]);
}

double CilkSclMlt(const double* A, const double* B, const int len){
	double result = 0;
	for(int k = 0; k < len; k++){
		result = result + A[k]*B[k];
	}
	return result;
}

void ParallelCilkMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w){
	double* tr = new double [src1w * src2w];
	ParallelCilkTranspose(src2, tr, src2w, src1w);
	cilk_for (int i = 0; i < src1h; i++){
		for (int j = 0; j < src2w; j++){
			double *vec1, *vec2;
			vec1 = &(src1[i*src1w]);
			vec2 = &(tr[j*src1w]);
			dst[i * src2w + j] = CilkSclMlt(vec1, vec2, src1w);
		}
	}
	delete [] tr;
}

void ParallelCilk2MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w){
	double* tr = new double [src1w * src2w];
	ParallelCilkTranspose(src2, tr, src2w, src1w);
	cilk_for (int i = 0; i < src1h; i++){
		cilk_for (int j = 0; j < src2w; j++){
			double *vec1, *vec2;
			vec1 = &(src1[i*src1w]);
			vec2 = &(tr[j*src1w]);
			dst[i * src2w + j] = CilkSclMlt(vec1, vec2, src1w);
		}
	}
	delete [] tr;
}

void ParallelCilk3MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w){
	double* tr = new double [src1w * src2w];
	ParallelCilkTranspose(src2, tr, src2w, src1w);
	cilk_for (int i = 0; i < src1h; i++){
		cilk_for (int j = 0; j < src2w; j++){
			double *vec1, *vec2;
			vec1 = &(src1[i*src1w]);
			vec2 = &(tr[j*src1w]);
			dst[i * src2w + j] = ParallelCilkSclMlt(vec1, vec2, src1w);
		}
	}
	delete [] tr;
}

void ParallelCilkBlock1MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs){
	int blockSize = bs;
	if ((src1h < blockSize)||(src1w < blockSize)||(src2w < blockSize)){
		ParallelCilkMMult(src1, src2, dst, src1h, src1w, src2w);
	} else if ((src1h == blockSize)&&(src1w == blockSize)&&(src2w == blockSize)){
		ParallelCilkMMult(src1, src2, dst, src1h, src1w, src2w);
	} else {
		double *A00 = new double[blockSize * blockSize];
		double *A01 = new double[blockSize * (src1w - blockSize)];
		double *A10 = new double[(src1h - blockSize) * blockSize];
		double *A11 = new double[(src1h - blockSize) * (src1w - blockSize)];

		double *B00 = new double[blockSize * blockSize];
		double *B01 = new double[blockSize * (src2w - blockSize)];
		double *B10 = new double[(src1w - blockSize) * blockSize];
		double *B11 = new double[(src1w - blockSize) * (src2w - blockSize)];

		double *C00 = new double[blockSize * blockSize];
		double *C01 = new double[blockSize * (src2w - blockSize)];
		double *C10 = new double[(src1h - blockSize) * blockSize];
		double *C11 = new double[(src1h - blockSize) * (src2w - blockSize)];

		double *P1 = new double[blockSize * blockSize];
		double *P2 = new double[blockSize * blockSize];
		double *P3 = new double[blockSize * (src2w - blockSize)];
		double *P4 = new double[blockSize * (src2w - blockSize)];
		double *P5 = new double[(src1h - blockSize) * blockSize];
		double *P6 = new double[(src1h - blockSize) * blockSize];
		double *P7 = new double[(src1h - blockSize) * (src2w - blockSize)];
		double *P8 = new double[(src1h - blockSize) * (src2w - blockSize)];
		
		{
			cilk_for(int i = 0; i < blockSize; i++){
				for(int j = 0; j < blockSize; j++){
					A00[i * blockSize + j] = src1[i * src1w + j];
				}
				for(int j = blockSize; j < src1w; j++){
					A01[i * (src1w - blockSize) + (j - blockSize)] = src1[i * src1w + j];
				}
			}
			cilk_for(int i = blockSize; i < src1h; i++){
				for(int j = 0; j < blockSize; j++){
					A10[(i - blockSize) * blockSize + j] = src1[i * src1w + j];
				}
				for(int j = blockSize; j < src1w; j++){
					A11[(i - blockSize) * (src1w - blockSize) + (j - blockSize)] = src1[i * src1w + j];
				}
			}
		}

		{
			cilk_for(int i = 0; i < blockSize; i++){
				for(int j = 0; j < blockSize; j++){
					B00[i * blockSize + j] = src2[i * src2w + j];
				}
				for(int j = blockSize; j < src2w; j++){
					B01[i * (src2w - blockSize) + (j - blockSize)] = src2[i * src2w + j];
				}
			}
			cilk_for(int i = blockSize; i < src1w; i++){
				for(int j = 0; j < blockSize; j++){
					B10[(i - blockSize) * blockSize + j] = src2[i * src2w + j];
				}
				for(int j = blockSize; j < src2w; j++){
					B11[(i - blockSize) * (src2w - blockSize) + (j - blockSize)] = src2[i * src2w + j];
				}
			}
		}

		ParallelCilkBlock1MMult(A00, B00, P1, blockSize, blockSize, blockSize, blockSize);
		ParallelCilkBlock1MMult(A01, B10, P2, blockSize, (src1w - blockSize), blockSize, blockSize);
		ParallelCilkAdd(P1, P2, C00, blockSize, blockSize);
		//PrintMat(C00, blockSize, blockSize);

		ParallelCilkBlock1MMult(A00, B01, P3, blockSize, blockSize, (src2w - blockSize), blockSize);
		ParallelCilkBlock1MMult(A01, B11, P4, blockSize, (src1w - blockSize), (src2w - blockSize), blockSize);
		ParallelCilkAdd(P3, P4, C01, (src2w - blockSize), blockSize);
		//PrintMat(C01, (src2w - blockSize), blockSize);

		ParallelCilkBlock1MMult(A10, B00, P5, (src1h - blockSize), blockSize, blockSize, blockSize);
		ParallelCilkBlock1MMult(A11, B10, P6, (src1h - blockSize), (src1w - blockSize), blockSize, blockSize);
		ParallelCilkAdd(P5, P6, C10, blockSize, (src1h - blockSize));
		//PrintMat(C10, blockSize, (src1h - blockSize));

		ParallelCilkBlock1MMult(A10, B01, P7, (src1h - blockSize), blockSize, (src2w - blockSize), blockSize);
		ParallelCilkBlock1MMult(A11, B11, P8, (src1h - blockSize), (src1w - blockSize), (src2w - blockSize), blockSize);
		ParallelCilkAdd(P7, P8, C11, (src2w - blockSize), (src1h - blockSize));
		//PrintMat(C11, (src2w - blockSize), (src1h - blockSize));

		{
			cilk_for(int i = 0; i < blockSize; i++){
				for(int j = 0; j < blockSize; j++){
					dst[i * src2w + j] = C00[i * blockSize + j];
				}
				for(int j = blockSize; j < src2w; j++){
					dst[i * src2w + j] = C01[i * (src2w - blockSize) + (j - blockSize)];
				}
			}
			cilk_for(int i = blockSize; i < src1h; i++){
				for(int j = 0; j < blockSize; j++){
					dst[i * src2w + j] = C10[(i - blockSize) * blockSize + j];
				}
				for(int j = blockSize; j < src2w; j++){
					dst[i * src2w + j] = C11[(i - blockSize) * (src2w - blockSize) + (j - blockSize)];
				}
			}
		}

		delete[] A00;
		delete[] A01;
		delete[] A10;
		delete[] A11;

		delete[] B00;
		delete[] B01;
		delete[] B10;
		delete[] B11;

		delete[] C00;
		delete[] C01;
		delete[] C10;
		delete[] C11;

		delete[] P1;
		delete[] P2;
		delete[] P3;
		delete[] P4;
		delete[] P5;
		delete[] P6;
		delete[] P7;
		delete[] P8;

	}
}

void ParallelCilkBlock2MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs){
	int blockSize = bs;
	int blockCount1h = ceil((double)src1h / blockSize);
	int blockCount1w = ceil((double)src1w / blockSize);
	int blockCount2w = ceil((double)src2w / blockSize);

	double *tB = new double[src2w * src1w];
	ParallelCilkTranspose(src2, tB, src1w, src2w);

	{
		cilk_for(int i = 0; i < src1h; i++){
			for (int j = 0; j < src2w; j++){
				dst[i * src2w + j] = 0;
			}
		}
	}

	cilk_for (int ib = 0; ib < blockCount1h; ib++){
		for (int jb = 0; jb < blockCount2w; jb++){
			for (int kb = 0; kb < blockCount1w; kb++){
				int endi = min(((ib + 1) * blockSize), src1h);
				int endj = min(((jb + 1) * blockSize), src2w);
				int endk = min(((kb + 1) * blockSize), src1w);
				cilk_for (int i = (ib * blockSize); i < endi; i++){
					for (int j = (jb * blockSize); j < endj; j++){
						double *vec1, *vec2;
						int k = kb * blockSize;
						vec1 = &(src1[i * src1w + k]);
						vec2 = &(tB[j * src1w + k]);
						dst[i * src2w + j] += CilkSclMlt(vec1, vec2, endk - k);
					}
				}
			}
		}
	}

	delete[] tB;
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