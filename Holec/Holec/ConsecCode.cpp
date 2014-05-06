#include "stdafx.h"
#include "ConsecCode.h"

void PrintMat(double *A, int w, int h)
{
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			cout<<A[i*w+j]<<" ";
		}
		cout<<endl;
	}
}

void ConsecAdd(const double *A, const double *B, double *dst, const int w, const int h)
{
	int i, j;
	for(i = 0; i < h; i++){
		for(j = 0; j < w; j++){
			dst[i * w + j] = A[i * w + j] + B[i * w + j];
		}
	}
}

void ConsecAddEq(const double *A, double *dst, const int w, const int h)
{
	int i, j;
	for(i = 0; i < h; i++){
		for(j = 0; j < w; j++){
			dst[i * w + j] = dst[i * w + j] + A[i * w + j];
		}
	}
}


void ConsecTranspose(const double* src, double* dst, const int wN, const int hN){
	for (int i = 0; i < hN; i++){
        for (int j = 0; j < wN; j++){
            dst[i * wN + j] = src[j * hN + i];
		}
	}
}

double ConsecSclMlt(const double* A, const double* B, const int len){
	double result = 0;
	for(int k = 0; k < len; k++){
		result = result + A[k]*B[k];
	}
	return result;
}

void ConsecMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w){
	double* tr = new double [src1w * src2w];
	ConsecTranspose(src2, tr, src1w, src2w);
	for (int i = 0; i < src1h; i++){
		for (int j = 0; j < src2w; j++){
			double *vec1, *vec2;
			vec1 = &(src1[i*src1w]);
			vec2 = &(tr[j*src1w]);
			dst[i*src2w + j] = ConsecSclMlt(vec1, vec2, src1w);
		}
	}
	delete [] tr;
}

void ConsecBlock1MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs){
	int blockSize = bs;
	if ((src1h < blockSize)||(src1w < blockSize)||(src2w < blockSize)){
		ConsecMMult(src1, src2, dst, src1h, src1w, src2w);
	} else if ((src1h == blockSize)&&(src1w == blockSize)&&(src2w == blockSize)){
		ConsecMMult(src1, src2, dst, src1h, src1w, src2w);
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
		
		for(int i = 0; i < blockSize; i++){
			for(int j = 0; j < blockSize; j++){
				A00[i * blockSize + j] = src1[i * src1w + j];
			}
			for(int j = blockSize; j < src1w; j++){
				A01[i * (src1w - blockSize) + (j - blockSize)] = src1[i * src1w + j];
			}
		}
		for(int i = blockSize; i < src1h; i++){
			for(int j = 0; j < blockSize; j++){
				A10[(i - blockSize) * blockSize + j] = src1[i * src1w + j];
			}
			for(int j = blockSize; j < src1w; j++){
				A11[(i - blockSize) * (src1w - blockSize) + (j - blockSize)] = src1[i * src1w + j];
			}
		}

		for(int i = 0; i < blockSize; i++){
			for(int j = 0; j < blockSize; j++){
				B00[i * blockSize + j] = src2[i * src2w + j];
			}
			for(int j = blockSize; j < src2w; j++){
				B01[i * (src2w - blockSize) + (j - blockSize)] = src2[i * src2w + j];
			}
		}
		for(int i = blockSize; i < src1w; i++){
			for(int j = 0; j < blockSize; j++){
				B10[(i - blockSize) * blockSize + j] = src2[i * src2w + j];
			}
			for(int j = blockSize; j < src2w; j++){
				B11[(i - blockSize) * (src2w - blockSize) + (j - blockSize)] = src2[i * src2w + j];
			}
		}

		ConsecBlock1MMult(A00, B00, P1, blockSize, blockSize, blockSize);
		ConsecBlock1MMult(A01, B10, P2, blockSize, (src1w - blockSize), blockSize);
		ConsecAdd(P1, P2, C00, blockSize, blockSize);
		//PrintMat(C00, blockSize, blockSize);

		ConsecBlock1MMult(A00, B01, P3, blockSize, blockSize, (src2w - blockSize));
		ConsecBlock1MMult(A01, B11, P4, blockSize, (src1w - blockSize), (src2w - blockSize));
		ConsecAdd(P3, P4, C01, (src2w - blockSize), blockSize);
		//PrintMat(C01, (src2w - blockSize), blockSize);

		ConsecBlock1MMult(A10, B00, P5, (src1h - blockSize), blockSize, blockSize);
		ConsecBlock1MMult(A11, B10, P6, (src1h - blockSize), (src1w - blockSize), blockSize);
		ConsecAdd(P5, P6, C10, blockSize, (src1h - blockSize));
		//PrintMat(C10, blockSize, (src1h - blockSize));

		ConsecBlock1MMult(A10, B01, P7, (src1h - blockSize), blockSize, (src2w - blockSize));
		ConsecBlock1MMult(A11, B11, P8, (src1h - blockSize), (src1w - blockSize), (src2w - blockSize));
		ConsecAdd(P7, P8, C11, (src2w - blockSize), (src1h - blockSize));
		//PrintMat(C11, (src2w - blockSize), (src1h - blockSize));

		for(int i = 0; i < blockSize; i++){
			for(int j = 0; j < blockSize; j++){
				dst[i * src2w + j] = C00[i * blockSize + j];
			}
			for(int j = blockSize; j < src2w; j++){
				dst[i * src2w + j] = C01[i * (src2w - blockSize) + (j - blockSize)];
			}
		}
		for(int i = blockSize; i < src1h; i++){
			for(int j = 0; j < blockSize; j++){
				dst[i * src2w + j] = C10[(i - blockSize) * blockSize + j];
			}
			for(int j = blockSize; j < src2w; j++){
				dst[i * src2w + j] = C11[(i - blockSize) * (src2w - blockSize) + (j - blockSize)];
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

void ConsecBlock2MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs){
	int blockSize = bs;
	int blockCount1h = ceil((double)src1h / blockSize);
	int blockCount1w = ceil((double)src1w / blockSize);
	int blockCount2w = ceil((double)src2w / blockSize);

	double *tB = new double[src2w * src1w];
	ConsecTranspose(src2, tB, src1w, src2w);

	for(int i = 0; i < src1h; i++){
		for (int j = 0; j < src2w; j++){
			dst[i * src2w + j] = 0;
		}
	}

	int ib, jb, kb;
	for (ib = 0; ib < blockCount1h; ib++){
		for (jb = 0; jb < blockCount2w; jb++){
			for (kb = 0; kb < blockCount1w; kb++){
				int endi = min(((ib + 1) * blockSize), src1h);
				int endj = min(((jb + 1) * blockSize), src2w);
				int endk = min(((kb + 1) * blockSize), src1w);
				for (int i = (ib * blockSize); i < endi; i++){
					for (int j = (jb * blockSize); j < endj; j++){
						for (int k = (kb * blockSize); k < endk; k++){
							dst[i * src2w + j] += src1[i * src1w + k] * tB[j * src1w + k];
						}
					}
				}
			}
		}
	}

	delete[] tB;
}

void ConsecBlock3MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs){
	int blockSize = bs;
	int blockCount1h = ceil((double)src1h / blockSize);
	int blockCount1w = ceil((double)src1w / blockSize);
	int blockCount2w = ceil((double)src2w / blockSize);

	for(int i = 0; i < src1h; i++){
		for (int j = 0; j < src2w; j++){
			dst[i * src2w + j] = 0;
		}
	}

	int ib, jb, kb;
	for (ib = 0; ib < blockCount1h; ib++){
		for (jb = 0; jb < blockCount2w; jb++){
			for (kb = 0; kb < blockCount1w; kb++){
				int endi = min(((ib + 1) * blockSize), src1h);
				int endj = min(((jb + 1) * blockSize), src2w);
				int endk = min(((kb + 1) * blockSize), src1w);
				for (int i = (ib * blockSize); i < endi; i++){
					for (int k = (kb * blockSize); k < endk; k++){
						for (int j = (jb * blockSize); j < endj; j++){
							dst[i * src2w + j] += src1[i * src1w + k] * src2[k * src2w + j];
						}
					}
				}
			}
		}
	}
}

void ConsecHolec(double* A, int n){
	//Result writes to source matrix A
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

void Consec1Holec(double* src, double* dst, int n){
	//Zeros element (j, i) writes when calculate element (i, j)
	for(int i = 0; i < n; i++){
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

void Consec2Holec(double* src, double* dst, int n){
	//Zeros elements (i, i+1), ... , (i, n) writes after calculate elements (i, 1), ... , (i, i)
	for(int i = 0; i < n; i++){
		double sum = 0;
		for (int j = 0; j < i; j++){
			sum = 0;
			for (int k = 0; k < j; k++){
				sum += dst[i*n+k]*dst[j*n+k];
			}
			dst[i*n+j] = (src[i*n+j]-sum)/dst[j*n+j];
		}
		sum = 0;
		for (int k = 0; k < i; k++){
			sum += dst[i*n+k]*dst[i*n+k];
		}
		dst[i*n+i] = sqrt(src[i*n+i] - sum);
		for (int j = i+1; j < n; j++){
			dst[i*n+j] = 0;
		}
	}
}
