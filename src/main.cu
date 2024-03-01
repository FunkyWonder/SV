// #include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <algorithm>

// System includes
#include <stdio.h>
#include <random>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <curand.h>
#include <curand_kernel.h>

#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

// #include "../include/helper_cuda.h"
//  // helper functions and utilities to work with CUDA
//  #include <helper_functions.h>
//  #include <helper_cuda.h>

// #include <malloc.h>

#define ALF 1.0e-14 /*Ensures sufficient decrease in function value in lnsrch.*/
#define ITMAX 200	/*Maximum allowed number of iterations in dfpmin*/
#define EPS 1.0e-12
#define EPS1 1.0e-6
#define EPS2 1.0e-8
#define EPS3 1.0e-4
#define JMAX 25
#define TOLX (4 * EPS) /*Convergence criterion on x values.*/
#define STPMX 0.1	   /*Scaled maximum step length allowed in line searches.*/
#define FREEALL             \
	{                       \
		free(xi);           \
		free(pnew);         \
		freemat(hessin, n); \
		free(hdg);          \
		free(g);            \
		free(dg);           \
	}
#define CON 1.4 /*Stepsize is decreased by CON at each iteration.*/
#define CON2 (CON * CON)
#define BIG 1.0e30
#define SAFE 2.0 /*Return when error is SAFE worse than the best so far.*/
#define pdim 4
#define nb_fils_max 1
#define SWAP(a, b) \
	temp = (a);    \
	(a) = (b);     \
	(b) = temp;

const double miPi = 3.141592653589793238462643;
const double ctrunceva = 1.0;
const double ctrunc = 3.5;
const int B = 1000;
// const int l = 24050;
const int l = 100;

typedef unsigned int uint32;
typedef signed int int32;

static uint32 jz, jsr = 123456789;

int lags, invcont, kbar, contS, seed;
double D1p, pd1, logsqrt2pi, lbound, phiconst, csqtr, m0true, gammakbtrue, btrue, sigmadtrue, alphaaux, sqrt2dPi;
double *driftmut, *monthmut, *shortmut, *ma50t, *ma250t;
double **Weightmatrix;

static int32 hz;
static uint32 iz, kn[128], ke[256];
static float wn[128], fn[128], we[256], fe[256];

/********* Pdf of univariate normal *************/
__host__ __device__ double pdf_norm(double x, double mu, double sigma)
{
	double val;

	val = (x - mu) / sigma;
	val = -pow(val, 2.0) / 2;
	val = exp(val) / (sqrt(2 * M_PI) * sigma);

	if (fabs(val) > 0.0)
		return val;
	else
		return 0.0;
}

__host__ __device__ double PhiErf(double x)
{
	double y;

	y = 0.5 * (1.0 + erf(x / M_SQRT2));

	return y;
}
__host__ __device__ double FqCauchy(double x)
{
	double val;

	val = (x / (2 * (1.0 + pow(x, 2.0))) + atan(x) / 2 + M_PI / 4) / (M_PI / 2);

	return val;
}

#define A 12
__host__ __device__ int compare(int a, int b, double *hsV)
{
	if (hsV[a] > hsV[b])
		return 1;
	else if (hsV[a] < hsV[b])
		return -1;
	else
		return 0;
}

struct DMatrix
{
	double *ptr;
	int w;
	int h;

	__device__ __host__ double *cell(int x, int y)
	{
		return &ptr[y * this->w + x];
	}

	__device__ __host__ double *row(int y)
	{
		return this->ptr + (this->w * y);
	}

	__device__ __host__ void free()
	{
		cudaFree(this->ptr);
	}
};

struct FMatrix
{
	float *ptr;
	int w;
	int h;

	float *cell(int x, int y)
	{
		return &ptr[y * this->w + x];
	}

	void free()
	{
		cudaFree(this->ptr);
	}
};

__host__ __device__ double *allocvec(int n)
{
	/**
	 * @param n number of elements to allocate
	 * @return pointer to allocated array of n elements
	 */
	double *z;

	cudaMalloc((void **)&z, sizeof(double) * n);

	return z;
}

__device__ __host__
	DMatrix
	allocmat(int m, int n)
{
	/**
	 * @param m number of rows in the matrix
	 * @param n number of columns in the matrix
	 * @return pointer to allocated array of m elements
	 */
	double *z;

	cudaMalloc(&z, sizeof(double *) * m * n);

	return {
		z,
		m,
		n};
}

__device__ __host__ int *alocintvec(int n)
{
	/**
	 * @param n the number of elements in the vector
	 * @return a pointer to a vector of integers
	 */
	int *z;
	cudaMalloc(&z, sizeof(int) * n);

	return z;
}

__device__ int initv(double *U, double *V, int n)
{
	double *T, *W;
	int i;

	T = U;
	W = V;

	for (i = 0; i < n; i++) // Loop over vector elements
	{
		*T = *W;
		T++;
		W++;
	}

	return 1;
}

__device__ int prodcv(double *y1, double c, int n, double *res)
{
	/**
	 *@param y1 the vector of values
	 *@param c the constant
	 *@param n the length of the vector
	 *@param res the result vector
	 *@return 1 if success, 0 if failed
	 */
	int i;
	double *y, *z;

	y = y1;
	z = res;

	for (i = 0; i < n; i++)
	{
		(*z) = c * (*y);
		z++;
		y++;
	}

	return 1;
}

__device__ int sumvect(double *y1, double *y2, int n, double *res)
{
	int i;
	double *y, *w, *z;

	z = res;
	y = y1;
	w = y2;

	for (i = 0; i < n; i++)
	{
		(*z) = (*y) + (*w);
		z++;
		y++;
		w++;
	}

	return 1;
}

__device__
	DMatrix
	sortm(DMatrix P, int ds)
{
	printf("sortm start");

	int threadIndex = threadIdx.x + (blockDim.x + threadIdx.y);

	double *z;
	double Min, Max, Max2;
	int i, imin, imax, imax2;

	Min = *P.cell(0, ds);
	imin = 0;

	for (i = 1; i < ds + 1; i++)
		if (*P.cell(i, ds) < Min)
		{
			Min = *P.cell(i, ds);
			imin = i;
		}

	if (imin != 0)
	{
		z = P.row(imin);

		memcpy(P.row(imin), P.row(0), P.w);
		memcpy(P.row(0), z, P.w);
	}

	Max = *P.cell(ds, ds);
	imax = ds;

	for (i = 1; i < ds; i++)
		if (*P.cell(i, ds) > Max)
		{
			Max = *P.cell(i, ds);
			imax = i;
		}

	if (imax != ds)
	{
		memcpy(z, P.row(imax), P.w);
		memcpy(P.row(imax), P.row(ds), P.w);
		memcpy(P.row(ds), z, P.w);
	}

	Max2 = *P.cell(ds - 1, ds);
	imax2 = ds - 1;

	for (i = 1; i < ds - 1; i++)
		if (*P.cell(i, ds) > Max2)
		{
			Max2 = *P.cell(i, ds);
			imax2 = i;
		}
	if (imax2 != ds - 1)
	{
		memcpy(z, P.row(imax2), P.w);
		memcpy(P.row(imax2), P.row(ds - 1), P.w);
		memcpy(P.row(ds - 1), z, P.w);
	}

	return P;
}

__device__ double PF(double *x, double *rt, double *pt, double **z2, double **z3, curandState *randState, double *hsV)
{
	clock_t time = clock();

	int jz, jsr = 123456789;

	DMatrix hs = allocmat(B, 2);
	double *Probi = allocvec(B);
	double *piMalik = allocvec(B + 1);
	int *sigmacount = alocintvec(B);

	const double mu = x[0];
	const double phi = x[1];
	const double sigma = x[2];
	const double mud = x[3];

	double phi2 = pow(phi, 2.0);
	double sigma2 = sigma * sigma;

	for (int cont = 0; cont < B; cont++)
		// TODO: Was rnor between 0.0-1.0 like curand_uniform?
		//*hs.cell(cont, 1) = mu / (1.0 - phi) + sqrt(sigma2 / (1.0 - phi2)) * curand_uniform(randState);
		*hs.cell(cont, 1) = mu / (1.0 - phi) + sqrt(sigma2 / (1.0 - phi2)) * curand_uniform(randState);

	// hs[cont][1] = mu / (1.0 - phi) + sqrt(sigma2 / (1.0 - phi2)) * rnor();

	double Like = 0.0;

	int i0 = 0;

	for (int t = i0; t < l; t++)
	{
		printf(".");

		double val = 0.0;

		for (int i = 0; i < B; i++)
			sigmacount[i] = i;

		for (int cont = 0; cont < B; cont++)
		{

			/******** Step 1 *******************/

			double index = mu + phi * (*hs.cell(cont, 1)) + sigma * curand_uniform(randState);
			hsV[cont] = index;
			*hs.cell(cont, 1) = index;
			//			double index = hsV[cont] = *hs.cell(cont, 1) = mu + phi * (*hs.cell(cont, 1)) + sigma * curand_uniform(randState);

			double sigmaxt = exp(*hs.cell(cont, 1) / 2);

			double driftmu = mud;

			Probi[cont] = pdf_norm(rt[t], driftmu, sigmaxt);
		}

		val = 0.0;
		for (int cont = 0; cont < B; cont++)
			val += Probi[cont];

		thrust::sort(sigmacount, sigmacount + B, [=](int &x, int &y)
					 { return compare(x, y, hsV) == -1; });

		for (int cont = 0; cont < B; cont++)
		{
			hsV[cont] = *hs.cell(sigmacount[cont], 1);
			*hs.cell(cont, 0) = Probi[sigmacount[cont]];
		}

		for (int cont = 0; cont < B; cont++)
		{
			Probi[cont] = *hs.cell(cont, 0);
			*hs.cell(cont, 1) = hsV[cont];
		}

		if (fabs(val) > 0.0)
		{
			Like += log(val / B);

			for (int cont = 0; cont < B; cont++)
				Probi[cont] = Probi[cont] / val;

			piMalik[0] = Probi[0] / 2;
			piMalik[B] = Probi[B - 1] / 2;
			for (int cont = 1; cont < B; cont++)
				piMalik[cont] = (Probi[cont] + Probi[cont - 1]) / 2;

			/* Generating from the multinomial distribution, using Malmquist ordered statistics */

			int contres0 = B;
			double a = 1.0;
			for (int cont = 0; cont < B; cont++)
			{
				a = pow(curand_uniform(randState), 1.0 / contres0) * a;
				hsV[B - cont - 1] = a;
				contres0 = contres0 - 1;
			}

			double s = 0.0;
			int j = 0;
			for (int cont = 0; cont < (B + 1); cont++)
			{
				s += piMalik[cont];
				while ((hsV[j] <= s) && (j < B))
				{
					sigmacount[j] = cont;
					*hs.cell(j, 0) = (hsV[j] - (s - piMalik[cont])) / piMalik[cont];
					j = j + 1;
				}
			}

			for (int cont = 0; cont < B; cont++)
			{
				if (sigmacount[cont] == 0)
					hsV[cont] = *hs.cell(0, 1);
				else
				{
					if (sigmacount[cont] == B)
						hsV[cont] = *hs.cell(B - 1, 1);
					else
						hsV[cont] = (*hs.cell(sigmacount[cont], 1) - *hs.cell(sigmacount[cont] - 1, 1)) * *hs.cell(cont, 0) + *hs.cell(sigmacount[cont] - 1, 1);
				}
			}

			for (int cont = 0; cont < B; cont++)
				*hs.cell(cont, 1) = hsV[cont];
		}
		else
		{
			Like = -BIG;
			t = l + 1;
		}
	}

	hs.free();

	cudaFree(Probi);
	cudaFree(sigmacount);
	cudaFree(piMalik);

	// time = clock() - time;
	// double gpu_time_used = ((double)time) / 1000;
	// printf("PF took %f * 1000 cycles to execute \n", gpu_time_used);
	printf("PF completed");

	return -Like;
}

__global__ void nlminLvectSimplex(
	DMatrix x0, int n, double *lambda, double *yaux, double *epsin, double **At, double **yni, double epsilon, int dim, curandState *globalRands, double *hsV)
{
	int threadIndex = threadIdx.x + (blockDim.x + threadIdx.y);

	curand_init(0, threadIndex, 0, &globalRands[threadIndex]);

	curandState localState = globalRands[threadIndex];

	clock_t time;
	double cpu_time_used;

	printf("a");

	time = clock();

	printf("b");

	DMatrix Pf;

	double *G, *z, *Ptry, *Ptry2, *w, *vec;

	double ftry, ftry2, tol;

	int i, j, j1, k;

	G = allocvec(dim);
	z = allocvec(dim);
	Pf = allocmat(dim + 1, dim + 1);
	Ptry = allocvec(dim + 1);
	Ptry2 = allocvec(dim + 1);
	w = allocvec(dim);
	vec = allocvec(dim);

	printf("c");

	for (i = 0; i < dim + 1; i++)
		for (j = 0; j < dim; j++)
			*(Pf.cell(i, j)) = *x0.cell(threadIndex, j);

	for (i = 0; i < dim; i++)
		*(Pf.cell(i + 1, i)) = *(Pf.cell(i + 1, i)) + lambda[i];

	printf("d");

	for (i = 0; i < dim + 1; i++)
		*Pf.cell(i, dim) = PF(Pf.cell(i, 0), yaux, epsin, At, yni, &localState, hsV);

	k = 0;

	Pf = sortm(Pf, dim);

	printf("e");

	tol = 1.0;

	printf("1");

	for (k = 0; k < n; k++)
	{

		if (tol < epsilon)
		{
			k = 2 * n;
		}
		else
		{
			for (j = 0; j < dim; j++)
				G[j] = 0.0;
			for (i = 0; i < dim + 1; i++)
			{
				sumvect(G, Pf.row(i), dim, w);
				initv(G, w, dim);
			}
			prodcv(G, 1.0 / (dim + 1), dim, w);
			initv(G, w, dim);
			prodcv(Pf.row(dim), -1.0, dim, w);
			sumvect(G, w, dim, vec);
			prodcv(vec, 2.0 * (dim + 1) / dim, dim, w);
			sumvect(Pf.row(dim), w, dim, Ptry);
			ftry = PF(Ptry, yaux, epsin, At, yni, &localState, hsV);
			if (ftry < *Pf.cell(0, dim))
			{
				prodcv(vec, 1.0 * (dim + 1) / dim, dim, w);
				sumvect(Ptry, w, dim, Ptry2);
				ftry2 = PF(Ptry2, yaux, epsin, At, yni, &localState, hsV);
				if (ftry2 < ftry)
				{
					for (j = 0; j < dim; j++)
						*Pf.cell(dim, j) = Ptry2[j];
					*Pf.cell(dim, dim) = ftry2;
				}
				else
				{
					for (j = 0; j < dim; j++)
						*Pf.cell(dim, j) = Ptry[j];
					*Pf.cell(dim, dim) = ftry;
				}
			}
			else
			{
				if (ftry > *Pf.cell(dim - 1, dim))
				{
					prodcv(vec, 0.5 * (dim + 1) / dim, dim, w);
					sumvect(Pf.row(dim), w, dim, Ptry);
					ftry = PF(Ptry, yaux, epsin, At, yni, &localState, hsV);
					if (ftry > *Pf.cell(dim, dim))
					{
						for (j = 1; j < dim + 1; j++)
						{
							sumvect(Pf.row(0), Pf.row(j), dim, w);
							prodcv(w, 0.5, dim, z);
							for (j1 = 0; j1 < dim; j1++)
								*Pf.cell(j, j1) = z[j1];
							*Pf.cell(j, dim) = PF(Pf.row(j), yaux, epsin, At, yni, &localState, hsV);
						}
					}
					else
					{
						for (j = 0; j < dim; j++)
							*Pf.cell(dim, j) = Ptry[j];
						*Pf.cell(dim, dim) = ftry;
					}
				}
				else
				{
					for (j = 0; j < dim; j++)
						*Pf.cell(dim, j) = Ptry[j];
					*Pf.cell(dim, dim) = ftry;
				}
			}
			Pf = sortm(Pf, dim);
		}
		/*  tol=2*(Pf[dim][dim]-Pf[0][dim])/(fabs(Pf[dim][dim])+fabs(Pf[0][dim])+EPS1); */
		tol = 0.0;
		for (j = 0; j < dim; j++)
			tol += 2.0 * fabs(*Pf.cell(dim, j) - *Pf.cell(0, j)) / (fabs(*Pf.cell(dim, j)) + fabs(*Pf.cell(0, j)) + EPS1);
	}

	if (k < (2 * n - 1))
	{
		/*	printf("NO CONVERGENCE IN %d ITERATIONS\n",n);
			Pf[0][dim]=BIG;*/
	}

	cudaFree(G);
	cudaFree(z);
	cudaFree(Ptry);
	cudaFree(Ptry2);
	cudaFree(w);
	cudaFree(vec);

	time = clock() - time;
	cpu_time_used = ((double)time) / CLOCKS_PER_SEC; // in seconds
	printf("nlminvectlsimplex took %f seconds to execute \n", cpu_time_used);
}

int main(void)
{
	int devID = 0;
	cudaDeviceProp props;

	int argc = 0;
	// // This will pick the best possible CUDA capable device
	cudaSetDevice(devID);

	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);

	// Get GPU information
	// checkCudaErrors(cudaGetDevice(&devID));
	// checkCudaErrors(cudaGetDeviceProperties(&props, devID));
	printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name,
		   props.major, props.minor);

	printf("printf() is called. Output:\n\n");

	int i, j, j1, k, n, t, cont, jcont, trials, itno;
	double ftry, ftry1, ftry2, tol, a, sigmad0, val, eps1, sig, minval;
	double *pt, *mus, *muss, *tetaCons, *lambda;
	double **Pf, **test, **StdAux;

	DMatrix rts, tetainit;

	n = 250;

	FILE *fp; // File pointer
	char s[30], outputFile[100];

	lags = 250; // Number of lags

	trials = 20; // Number of trials
	itno = 1;	 // Number of iterations

	sqrt2dPi = sqrt(2.0 / miPi);
	csqtr = sqrt(ctrunc);
	pd1 = exp(-ctrunc / 2) / sqrt(2 * miPi);
	D1p = pow(fabs(-csqtr), ctrunc) * pd1;
	phiconst = 1.0 - 2 * PhiErf(-csqtr);
	lbound = 1.0;
	logsqrt2pi = log(sqrt(2.0 * miPi));

	/*printf("phi=%.12f\n",phiconst);*/

	double *muaux = (double *)malloc(pdim * sizeof(double)); // Vector for storing the auxiliary variables.
	tetaCons = (double *)malloc(pdim * sizeof(double));		 // Vector for storing the constraint variables.
	tetainit = allocmat((itno * trials), pdim);				 // Matrix for storing the initial constraint variables.

	DMatrix tetainitHost = {
		(double *)malloc(itno * trials * pdim * sizeof(double)),
		itno * trials,
		pdim};

	DMatrix rtsHost = {
		(double *)malloc(itno * l * sizeof(double)),
		itno,
		l};

	rts = allocmat(itno, l);   // Matrix for storing the time series. 1 by 24050
	pt = allocvec(l);		   // Vector for storing the time series.
	double *hsV = allocvec(B); // Vector for storing the Hessian values.
	lambda = allocvec(pdim);   // Vector for storing the lambda values.

	double *lambdaHost = (double *)malloc(pdim * sizeof(double));

	mus = allocvec(B);
	muss = allocvec(B);
	monthmut = allocvec(l); // Vector for storing the monthly mutation rates.
	shortmut = allocvec(l); // Vector for storing the short-term mutation rates.
	driftmut = allocvec(l); // Vector for storing the drift mutation rates.
	ma50t = allocvec(l);	// Vector for storing the moving average over 50 days.
	ma250t = allocvec(l);	// Vector for storing the moving average over 250 days.

	muaux[0] = -0.190749834;
	muaux[1] = 0.977385697;
	muaux[2] = 0.215634224;
	muaux[3] = 0.015011035;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1); // uniform distribution between 0 and 1

	for (i = 0; i < (itno * trials); i++) // Loops over the trials times itno.
	{
		for (j = 0; j < pdim; j++) // pdim=4, trials=20, and itno=1
			// For each trial, fill a vector with "muaux[j] * (0.99 + 0.02 * uni())" (uses ziggurat)
			*tetainitHost.cell(i, j) = muaux[j] * (0.99 + 0.02 * dis(gen));
	}

	fp = fopen("GSPC19280104-20230929.txt", "r");

	j = 0;

	// Stores GSPC19280104-20230929.txt data in rts.
	while (fgets(s, 30, fp) != NULL) // Reads the data from the file line by line. Stops when the end of the file is reached.
	{
		// if (j >= 0)				 // This is not needed.
		//*rtsHost.cell(0, j) = atof(s); // Fill the first column of rts with the data of the line converted to float.
		j++; // Increments the rows counter.
	}
	fclose(fp);

	sprintf(outputFile, "SV-T%dN%d_out.txt", l, B); // Generates the output files with enough lines

	fp = fopen(outputFile, "a");
	fprintf(fp, "%s	%s	%s	%s	%s\n", "a", "b", "sigma", "mu", "val"); // Sets "a b sigma mu val" in top row of the file
	fclose(fp);

	lambdaHost[0] = -0.5;
	lambdaHost[1] = -0.02;
	lambdaHost[2] = 0.2;
	lambdaHost[3] = 0.01;

	cudaMemcpy(lambda, lambdaHost, pdim, cudaMemcpyHostToDevice);
	free(lambdaHost);

	minval = BIG; // Big heeft een waarde van 1.0e30
	for (i = 0; i < pdim; i++)
		tetaCons[i] = *tetainitHost.cell(0, i); // tetaCons is een vector van 4, tetainit is een matrix van 4x20

	cudaMemcpy(tetainit.ptr, tetainitHost.ptr, tetainitHost.w * tetainitHost.h, cudaMemcpyHostToDevice);

	// Main loop

	curandState *states;
	cudaMalloc(&states, sizeof(curandState) * trials);

	// nlminLvectSimplex<<<1, trials >>>(tetainit, 2000, lambda, rts.row(0), pt, Weightmatrix, Weightmatrix, EPS1, pdim, states, hsV);
	nlminLvectSimplex<<<10, 100>>>(tetainit, 2000, lambda, rts.row(0), pt, Weightmatrix, Weightmatrix, EPS1, pdim, states, hsV);

	// test[0][2] = fabs(test[0][2]);

	// if (test[0][pdim] < minval)
	//{
	//	for (i = 0; i < pdim; i++)
	//		tetaCons[i] = test[0][i];
	//	minval = test[0][pdim];
	// }

	//// This part of the code writes the results to a file.
	// fp = fopen(outputFile, "a");
	// for (i = 0; i < pdim; i++)
	//	fprintf(fp, "%.12f	", test[0][i]); // Format the results to a fixed precision.
	// fprintf(fp, "%.16f\n", test[0][pdim]);
	// fclose(fp);

	///*	printf("jcont=%d\n",jcont);
	//	writemat("t",pdim+1,pdim+1,test);*/

	// freemat(test, pdim + 1);

	// End main loop

	// This part of the code frees the memory allocated for the matrices and vectors.
	free(muaux);
	rts.free();
	free(tetaCons);
	tetainit.free();
	cudaFree(hsV);
	cudaFree(lambda);
	cudaFree(mus);
	cudaFree(muss);
	cudaFree(pt);
	/*free(monthmut);
	free(shortmut);
	free(driftmut);
	free(ma50t);
	free(ma250t);*/

	exit(0);
}
