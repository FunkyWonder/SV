#pragma once

// #include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <algorithm>
#include <cstdint>

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
#include <cuda_fp16.h>

#define EPS 1.0e-12
#define EPS1 1.0e-6
#define CON 1.4 /*Stepsize is decreased by CON at each iteration.*/
#define BIG 1.0e30
#define pdim 4

#define fpxx half

const int B = 1000;
const int l = 24050;// 24050

typedef unsigned int uint32;
typedef signed int int32;

/********* Pdf of univariate normal *************/
__device__ __host__ double pdf_norm(fpxx x, fpxx mu, fpxx sigma)
{
	fpxx val;

	val = (x - mu) / sigma;
	val = -pow(val, 2.0) / 2;
	val = exp(val) / (sqrt(2 * M_PI) * sigma);

	if (__habs(val) > 0.0)
		return val;
	else
		return 0.0;
}

// __host__ __device__ double PhiErf(double x)
// {
// 	double y;

// 	y = 0.5 * (1.0 + erf(x / M_SQRT2));

// 	return y;
// }
// __host__ __device__ double FqCauchy(double x)
// {
// 	double val;

// 	val = (x / (2 * (1.0 + pow(x, 2.0))) + atan(x) / 2 + M_PI / 4) / (M_PI / 2);

// 	return val;
// }

__device__ __host__ int compare(int a, int b, fpxx *hsV)
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
	fpxx *ptr;
	int w;
	int h;

	__device__ __host__ fpxx *cell(int x, int y)
	{
		return &this->ptr[(this->w * y) + x];
	}

	__device__ __host__ fpxx *row(int y)
	{
		return &this->ptr[this->w * y];
	}

	__device__ __host__ void free()
	{
		cudaFree(this->ptr);
	}
};


__host__ __device__ fpxx *allocvec(int n)
{
	/**
	 * @param n number of elements to allocate
	 * @return pointer to allocated array of n elements
	 */
	fpxx *z;

	cudaMalloc((void **)&z, sizeof(fpxx) * n);

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
	fpxx *z;

	cudaMalloc(&z, sizeof(fpxx) * m * n);

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

__device__ __host__ int initv(fpxx *U, fpxx *V, int n)
{
	fpxx *T, *W;
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

__device__ __host__ int prodcv(fpxx *y1, fpxx c, int n, fpxx *res)
{
	/**
	 *@param y1 the vector of values
	 *@param c the constant
	 *@param n the length of the vector
	 *@param res the result vector
	 *@return 1 if success, 0 if failed
	 */
	int i;
	fpxx *y, *z;

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

__device__ __host__ int sumvect(fpxx *y1, fpxx *y2, int n, fpxx *res)
{
	int i;
	fpxx *y, *w, *z;

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

__device__ __host__
	DMatrix
	inline sortm(DMatrix P, int ds)
{
	int threadIndex = threadIdx.x + (blockDim.x + threadIdx.y);

	fpxx *z;
	fpxx Min, Max, Max2;
	int i, imin, imax, imax2;
	Min = *P.cell(0, ds);
	imin = 0;

	for (i = 1; i < ds + 1; i++) {
		if (*P.cell(i, ds) < Min)
		{
			Min = *P.cell(i, ds);
			imin = i;
		}
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

__device__ fpxx PF(fpxx *x, fpxx *rt, fpxx *pt, curandState *randState, fpxx *hsV)
{
	printf("pf start");

	DMatrix hs = allocmat(B, 2);
	fpxx *Probi = allocvec(B);
	fpxx *piMalik = allocvec(B + 1);
	int *sigmacount = alocintvec(B);

	const fpxx mu = x[0];
	const fpxx phi = x[1];
	const fpxx sigma = x[2];
	const fpxx mud = x[3];

	fpxx phi2 = pow(phi, 2.0);
	fpxx sigma2 = sigma * sigma;

	for (int cont = 0; cont < B; cont++)
		// TODO: Was rnor between 0.0-1.0 like curand_uniform?
		//*hs.cell(cont, 1) = mu / (1.0 - phi) + sqrt(sigma2 / (1.0 - phi2)) * curand_uniform(randState);
		*hs.cell(cont, 1) = mu / (1.0 - phi) + sqrt(sigma2 / (1.0 - phi2)) * curand_uniform(randState);

	// hs[cont][1] = mu / (1.0 - phi) + sqrt(sigma2 / (1.0 - phi2)) * rnor();

	fpxx Like = 0.0;

	int i0 = 0;

	for (int t = i0; t < l; t++)
	{
		// printf("%f\n", Like);
		printf("%d\n", t);

		fpxx val = 0.0;

		for (int i = 0; i < B; i++)
			sigmacount[i] = i;

		for (int cont = 0; cont < B; cont++)
		{

			/******** Step 1 *******************/

			fpxx index = mu + phi * (*hs.cell(cont, 1)) + sigma * curand_uniform(randState);
			hsV[cont] = index;
			*hs.cell(cont, 1) = index;
			//			fpxx index = hsV[cont] = *hs.cell(cont, 1) = mu + phi * (*hs.cell(cont, 1)) + sigma * curand_uniform(randState);

			fpxx sigmaxt = exp(*hs.cell(cont, 1) / 2);

			fpxx driftmu = mud;

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
			fpxx a = 1.0;
			for (int cont = 0; cont < B; cont++)
			{
				a = pow(curand_uniform(randState), 1.0 / contres0) * a;
				hsV[B - cont - 1] = a;
				contres0 = contres0 - 1;
			}

			fpxx s = 0.0;
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
	// fpxx gpu_time_used = ((fpxx)time) / 1000;
	// printf("PF took %f * 1000 cycles to execute \n", gpu_time_used);
	printf("PF completed");

	return -Like;
}


