// Holec.cpp: ���������� ����� ����� ��� ����������� ����������.
//

#include "stdafx.h"
#include "ConsecCode.h"
//#include "OpenMPCode.h"
//#include "CilkCode.h"
//#include "TBBCode.h"

#define DELTA (0.001)

void Init(double *A, int w, int h){
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			A[i*w+j]=rand()%5;
		}
	}
}

void RandomInit(double *A, int w, int h){
	double sum = 0;
	for(int i=0; i<h; i++){
		sum = 0;
		for(int j=i; j<w; j++){
			double val = rand()%5;
			A[i*w+j] = val;
			A[j*w+i] = val;
		}
		for (int j = 0; j < w; j++){
			sum += fabs(A[i*w+j]);
		}
		sum -= fabs(A[i*w+i]);
		A[i*w+i] = sum + (rand()%5);
	}
}

void LInit(double *A, int w, int h){
	for(int i=0; i<h; i++){
		for(int j=0; j<(i+1); j++){
			A[i*w+j]=rand()%5;
		}
		for(int j=(i+1); j<w; j++){
			A[i*w+j]=0;
		}
	}
}

void InitTest1(double *A, int w, int h){
    A[0] = 1; A[1] = 1; A[2] = 0;
	A[3] = 1; A[4] = 2; A[5] = 0;
	A[6] = 0; A[7] = 0; A[8] = 1;
}

void InitTest2(double *A, int w, int h){
    A[0] = 2; A[1] = -1; A[2] = 0;
	A[3] = -1; A[4] = 2; A[5] = -1;
	A[6] = 0; A[7] = -1; A[8] = 2;
}

void Print(double *A, int w, int h){
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			cout<<A[i*w+j]<<" ";
		}
		cout<<endl;
	}
}

bool CompareMatrix(double *gC, double *C, int size, double delta){
    for(int i=0; i<size; i++)
        if(fabs(gC[i]-C[i])>delta)
            return false;
    return true;
}

int main(int argc, char* argv[])
{
    //double *A, *B, *gC, *C;
	double *A, *B, *X, *gX;
    int seed = 666;
    bool rez = false;
    int n = 0;
    double s, f;

    if(argc < 3)
    {
        printf("Error!!!\nUsage:\n");
        printf("  <appName> <N> <seed>\n");    
        return 0;
    }

	//n = atoi(argv[1]);
	seed = atoi(argv[2]);  
	int n1 = 50;
	int n2 = 3 * n1;

	srand(seed);

	/*
    A=new double[n*n];
	B=new double[n*n];
	C=new double[n*n]; 
    gC=new double[n*n]; 
	*/
	A = new double[n1 * n1];
	B = new double[n1 * n2];
	X = new double[n1 * n2];
	gX = new double[n1 * n2];

	LInit(A, n1, n1);
	Init(B, n2, n1);

	/*
    //InitTest1(A, n, n);
	//RandomInit(A, n, n);
	//RandomInit(B, n, n);
	Init(A, n, n);
	Init(B, n, n);
	//Print(A, n, n);
	*/

	/*
	for (int bs = 20; bs < 520; bs+=20){
		double *C=new double[n*n];

		double s2 = omp_get_wtime();
		ConsecBlock2MMult(A, B, C, n, n, n, bs);
		double f2 = omp_get_wtime();
		
		cout<<bs<<"    "<<(f2 - s2)<<endl;
		delete[] C;
	}
	*/

	s = omp_get_wtime();
	ConsecSolve1(A, X, B, n1, n2);
    f = omp_get_wtime();
    printf("Execution time of 1 solving system: %lf\n", f-s);

	s = omp_get_wtime();
	ConsecSolve2(A, gX, B, n1, n2);
    f = omp_get_wtime();
    printf("Execution time of 2 solving system: %lf\n", f-s);

    s = omp_get_wtime();
	//ConsecMMult(A, B, C, n, n, n);
    f = omp_get_wtime();
	//Print(C, n, n);
    printf("Execution time of consec mlt: %lf\n", f-s);

	s = omp_get_wtime();
	//ConsecBlock1MMult(A, B, C, n, n, n, 1000);
    f = omp_get_wtime();
    printf("Execution time of consec block 1 mlt: %lf\n", f-s);

	s = omp_get_wtime();
	//ConsecBlock2MMult(A, B, C, n, n, n);
    f = omp_get_wtime();
    printf("Execution time of consec block 2 mlt: %lf\n", f-s);

	s = omp_get_wtime();
	//ParallelOMPBlock1MMult(A, B, C, n, n, n, 1000);
    f = omp_get_wtime();
    printf("Execution time of parallel omp block 1 mlt: %lf\n", f-s);

	s = omp_get_wtime();
	//ParallelOMPBlock2MMult(A, B, C, n, n, n);
    f = omp_get_wtime();
    printf("Execution time of parallel omp block 2 mlt: %lf\n", f-s);

	s = omp_get_wtime();
	//ParallelCilkBlock1MMult(A, B, C, n, n, n, 1000);
    f = omp_get_wtime();
    printf("Execution time of parallel cilk block 1 mlt: %lf\n", f-s);

	s = omp_get_wtime();
	//ParallelCilkBlock2MMult(A, B, gC, n, n, n);
    f = omp_get_wtime();
    printf("Execution time of parallel cilk block 2 mlt: %lf\n", f-s);

	s = omp_get_wtime();
	//ParallelTBBMMult(A, B, C, n, n, n);
    f = omp_get_wtime();
    printf("Execution time of parallel tbb mlt: %lf\n", f-s);

	s = omp_get_wtime();
	//ParallelTBBBlock1MMult(A, B, gC, n, n, n);
    f = omp_get_wtime();
    printf("Execution time of parallel tbb block 1 mlt: %lf\n", f-s);

	s = omp_get_wtime();
	//ParallelTBBBlock2MMult(A, B, gC, n, n, n);
    f = omp_get_wtime();
    printf("Execution time of parallel tbb block 2 mlt: %lf\n", f-s);

	rez=CompareMatrix(gX, X, n1 * n2, DELTA);
	printf("Compare: %i\n",rez);


    delete[] A;
	delete[] B;
	delete[] X;
	delete[] gX;
	//delete[] C;
	//delete[] gC;
    return 0;
} 
