#include "stdafx.h"
#include "OpenMPCode.h"`

void ParallelOMPAdd(const double *A, const double *B, double *dst, const int w, const int h)
{
	int i, j;
	# pragma omp parallel for private (j, i) schedule (dynamic)
	for(i = 0; i < h; i++){
		# pragma ivdep
		# pragma simd
		for(j = 0; j < w; j++){
			dst[i * w + j] = A[i * w + j] + B[i * w + j];
		}
	}
}

void ParallelOMPAddEq(const double *A, double *dst, const int w, const int h)
{
	int i, j;
	# pragma omp parallel for private (j, i) schedule (dynamic)
	for(i = 0; i < h; i++){
		# pragma ivdep
		# pragma simd
		for(j = 0; j < w; j++){
			dst[i * w + j] = dst[i * w + j] + A[i * w + j];
		}
	}
}


void ParallelOMPTranspose(const double* src, double* dst, const int wN, const int hN){
	int i, j;
	# pragma omp parallel for shared(src, dst) private(i, j) schedule (dynamic)
	for (i = 0; i < hN; i++){
		# pragma ivdep
		# pragma simd
        for (j = 0; j < wN; j++){
            dst[i * wN + j] = src[j * hN + i];
		}
	}
}

double ParallelOMPSclMlt(const double* A, const double* B, const int len){
	double result = 0;
	# pragma ivdep
	# pragma simd
	for(int k = 0; k < len; k++){
		result += A[k]*B[k];
	}
	return result;
}

void ParallelOMPMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w){
	double* tr = new double [src1w * src2w];
	ParallelOMPTranspose(src2, tr, src1w, src2w);
	int i, j, k;
	//double sum = 0;
	//omp_set_nested(true);
	#pragma omp parallel for shared(src1, tr, dst, src1h, src1w, src2w) private(i, j, k) schedule(dynamic) //reduction(+:sum)
	for (i = 0; i < src1h; i++){
		//#pragma omp parallel for shared(src1, src2, dst, src1h, src1w, src2w) private(j, k, sum) schedule(dynamic) //reduction(+:sum)
		//# pragma ivdep
		# pragma simd
		for (j = 0; j < src2w; j++){
			double *vec1, *vec2;
			vec1 = &(src1[i*src1w]);
			vec2 = &(tr[j*src1w]);
			dst[i*src2w + j] = ParallelOMPSclMlt(vec1, vec2, src1w);
		}
	}
	delete [] tr;
}

void ParallelOMPBlock1MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs){
	int blockSize = bs;
	if ((src1h < blockSize)||(src1w < blockSize)||(src2w < blockSize)){
		ParallelOMPMMult(src1, src2, dst, src1h, src1w, src2w);
	} else if ((src1h == blockSize)&&(src1w == blockSize)&&(src2w == blockSize)){
		ParallelOMPMMult(src1, src2, dst, src1h, src1w, src2w);
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
			int i, j;
			# pragma omp parallel for shared(src1, A00, A01, A10, A11) private(i, j) schedule(dynamic)
			for(i = 0; i < blockSize; i++){
				# pragma ivdep
				# pragma simd
				for(j = 0; j < blockSize; j++){
					A00[i * blockSize + j] = src1[i * src1w + j];
				}
				# pragma ivdep
				# pragma simd
				for(j = blockSize; j < src1w; j++){
					A01[i * (src1w - blockSize) + (j - blockSize)] = src1[i * src1w + j];
				}
			}
			for(i = blockSize; i < src1h; i++){
				# pragma ivdep
				# pragma simd
				for(j = 0; j < blockSize; j++){
					A10[(i - blockSize) * blockSize + j] = src1[i * src1w + j];
				}
				# pragma ivdep
				# pragma simd
				for(j = blockSize; j < src1w; j++){
					A11[(i - blockSize) * (src1w - blockSize) + (j - blockSize)] = src1[i * src1w + j];
				}
			}
		}

		{
			int i, j;
			# pragma omp parallel for shared(src2, B00, B01, B10, B11) private(i, j) schedule(dynamic)
			for(i = 0; i < blockSize; i++){
				# pragma ivdep
				# pragma simd
				for(j = 0; j < blockSize; j++){
					B00[i * blockSize + j] = src2[i * src2w + j];
				}
				# pragma ivdep
				# pragma simd
				for(j = blockSize; j < src2w; j++){
					B01[i * (src2w - blockSize) + (j - blockSize)] = src2[i * src2w + j];
				}
			}
			for(i = blockSize; i < src1w; i++){
				# pragma ivdep
				# pragma simd
				for(j = 0; j < blockSize; j++){
					B10[(i - blockSize) * blockSize + j] = src2[i * src2w + j];
				}
				# pragma ivdep
				# pragma simd
				for(j = blockSize; j < src2w; j++){
					B11[(i - blockSize) * (src2w - blockSize) + (j - blockSize)] = src2[i * src2w + j];
				}
			}
		}

		ParallelOMPBlock1MMult(A00, B00, P1, blockSize, blockSize, blockSize, blockSize);
		ParallelOMPBlock1MMult(A01, B10, P2, blockSize, (src1w - blockSize), blockSize, blockSize);
		ParallelOMPAdd(P1, P2, C00, blockSize, blockSize);
		//PrintMat(C00, blockSize, blockSize);

		ParallelOMPBlock1MMult(A00, B01, P3, blockSize, blockSize, (src2w - blockSize), blockSize);
		ParallelOMPBlock1MMult(A01, B11, P4, blockSize, (src1w - blockSize), (src2w - blockSize), blockSize);
		ParallelOMPAdd(P3, P4, C01, (src2w - blockSize), blockSize);
		//PrintMat(C01, (src2w - blockSize), blockSize);

		ParallelOMPBlock1MMult(A10, B00, P5, (src1h - blockSize), blockSize, blockSize, blockSize);
		ParallelOMPBlock1MMult(A11, B10, P6, (src1h - blockSize), (src1w - blockSize), blockSize, blockSize);
		ParallelOMPAdd(P5, P6, C10, blockSize, (src1h - blockSize));
		//PrintMat(C10, blockSize, (src1h - blockSize));

		ParallelOMPBlock1MMult(A10, B01, P7, (src1h - blockSize), blockSize, (src2w - blockSize), blockSize);
		ParallelOMPBlock1MMult(A11, B11, P8, (src1h - blockSize), (src1w - blockSize), (src2w - blockSize), blockSize);
		ParallelOMPAdd(P7, P8, C11, (src2w - blockSize), (src1h - blockSize));
		//PrintMat(C11, (src2w - blockSize), (src1h - blockSize));

		{
			int i, j;
			# pragma omp parallel for shared(dst, C00, C01, C10, C11) private(i, j) schedule(dynamic)
			for(i = 0; i < blockSize; i++){
				# pragma ivdep
				# pragma simd
				for(j = 0; j < blockSize; j++){
					dst[i * src2w + j] = C00[i * blockSize + j];
				}
				# pragma ivdep
				# pragma simd
				for(j = blockSize; j < src2w; j++){
					dst[i * src2w + j] = C01[i * (src2w - blockSize) + (j - blockSize)];
				}
			}
			for(i = blockSize; i < src1h; i++){
				# pragma ivdep
				# pragma simd
				for(j = 0; j < blockSize; j++){
					dst[i * src2w + j] = C10[(i - blockSize) * blockSize + j];
				}
				# pragma ivdep
				# pragma simd
				for(j = blockSize; j < src2w; j++){
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

void ParallelOMPBlock2MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs){
	int blockSize = bs;
	int blockCount1h = ceil((double)src1h / blockSize);
	int blockCount1w = ceil((double)src1w / blockSize);
	int blockCount2w = ceil((double)src2w / blockSize);

	double *tB = new double[src2w * src1w];
	ParallelOMPTranspose(src2, tB, src1w, src2w);

	//double s = omp_get_wtime();
	{
		int i, j;
		# pragma omp parallel for shared(dst) private(i, j) schedule(dynamic)
		for(i = 0; i < src1h; i++){
			# pragma ivdep
			# pragma simd
			for (j = 0; j < src2w; j++){
				dst[i * src2w + j] = 0;
			}
		}
	}
	//double f = omp_get_wtime();
	//cout<<"Time of zeros: "<<f-s<<endl;

	//omp_set_nested(true);
	//omp_set_num_threads(2);
	int ib, jb, kb;
	# pragma omp parallel for shared(dst, src1, tB) private(ib, jb, kb) schedule(dynamic)
	for (ib = 0; ib < blockCount1h; ib++){
		for (jb = 0; jb < blockCount2w; jb++){
			for (kb = 0; kb < blockCount1w; kb++){
				int endi = min(((ib + 1) * blockSize), src1h);
				int endj = min(((jb + 1) * blockSize), src2w);
				int endk = min(((kb + 1) * blockSize), src1w);
				int i, j, k;
				//omp_set_num_threads(2);
				# pragma omp parallel for shared(dst, src1, tB) private(i, j, k) schedule(dynamic)
				for (i = (ib * blockSize); i < endi; i++){
					for (j = (jb * blockSize); j < endj; j++){
						double *vec1, *vec2;
						int k = kb * blockSize;
						vec1 = &(src1[i * src1w + k]);
						vec2 = &(tB[j * src1w + k]);
						dst[i * src2w + j] += ParallelOMPSclMlt(vec1, vec2, endk - k);
					}
				}
			}
		}
	}

	delete[] tB;
}

void ParallelOMPBlock3MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs){
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
	# pragma omp parallel for shared(dst, src1, src2) private(ib, jb, kb) schedule(dynamic)
	for (ib = 0; ib < blockCount1h; ib++){
		for (jb = 0; jb < blockCount2w; jb++){
			# pragma ivdep
			# pragma simd
			for (kb = 0; kb < blockCount1w; kb++){
				int endi = min(((ib + 1) * blockSize), src1h);
				int endj = min(((jb + 1) * blockSize), src2w);
				int endk = min(((kb + 1) * blockSize), src1w);
				int i, j, k;
				//# pragma omp parallel for shared(dst, src1, src2) private(i, j, k) schedule(dynamic)
				for (i = (ib * blockSize); i < endi; i++){
					for (k = (kb * blockSize); k < endk; k++){
						# pragma ivdep
						# pragma simd
						for (j = (jb * blockSize); j < endj; j++){
							dst[i * src2w + j] += src1[i * src1w + k] * src2[k * src2w + j];
						}
					}
				}
			}
		}
	}
}

void ParallelOMP1Holec(double* A, int n){
	#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < n; i++){
		double sum = 0;
		//#pragma omp parallel for
		for (int j = 0; j < i; j++){
			sum = 0;
			//#pragma omp parallel for
			for (int k = 0; k < j; k++){
				sum += A[i*n+k]*A[j*n+k];
			}
			A[i*n+j] = (A[i*n+j]-sum)/A[j*n+j];
		}
		sum = 0;
		//#pragma omp parallel for
		for (int k = 0; k < i; k++){
			sum += A[i*n+k]*A[i*n+k];
		}
		A[i*n+i] = sqrt(A[i*n+i] - sum);
	}
}

void ParallelOMP1Holec(double* src, double* dst, int n){
	int i, j, k;
	#pragma simd
	#pragma omp parallel for ordered shared(src, dst, n) private(i, j, k) schedule(dynamic)
	for(i = 0; i < n; i++){
		double sum = 0;
		#pragma omp parallel for shared(src, dst, n) private(j, k) schedule(dynamic)
		for (j = 0; j < i; j++){
			sum = 0;
			//#pragma omp parallel for //schedule(dynamic)
			for (k = 0; k < j; k++){
				sum += dst[i*n+k]*dst[j*n+k];
			}
			dst[i*n+j] = (src[i*n+j]-sum)/dst[j*n+j];
			dst[j*n+i] = 0;
		}
		sum = 0;
		#pragma omp parallel for shared(src, dst, n) private(k) schedule(dynamic)
		for (k = 0; k < i; k++){
			sum += dst[i*n+k]*dst[i*n+k];
		}
		dst[i*n+i] = sqrt(src[i*n+i] - sum);
		/*
		#pragma omp parallel for shared(dst, n) private(j) schedule(dynamic)
		for (j = i+1; j < n; j++){
			dst[i*n+j] = 0;
		}
		*/
	}
	#pragma omp barrier
}


