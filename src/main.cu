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

// Include our own libraries
#include "main.h"

// #include "../include/helper_cuda.h"
//  // helper functions and utilities to work with CUDA
//  #include <helper_functions.h>
//  #include <helper_cuda.h>

// #include <malloc.h>


__global__ void nlminLvectSimplex(
	DMatrix x0, int n, fpxx *lambda, fpxx *yaux, fpxx *epsin, fpxx epsilon, int dim, curandState *globalRands, fpxx *hsV_global, DMatrix *Pf_out)
{
	int threadsPerBlock  = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;

	int threadIndex = blockNumInGrid * threadsPerBlock + threadNumInBlock;

	curand_init(0, threadIndex, 0, &globalRands[threadIndex]);

	fpxx *hsV = hsV_global + (threadIndex * B);

	curandState localState = globalRands[threadIndex];

	DMatrix Pf = allocmat(dim + 1, dim + 1);

	Pf_out[threadIndex] = Pf;

	fpxx *G, *z, *Ptry, *Ptry2, *w, *vec;

	fpxx ftry, ftry2, tol;

	int j1, k;

	G = allocvec(dim);
	z = allocvec(dim);
	Ptry = allocvec(dim + 1);
	Ptry2 = allocvec(dim + 1);
	w = allocvec(dim);
	vec = allocvec(dim);

	// for(int j = 0; j < pdim; j++) {
		// printf("%f ", __half2float(*x0.cell(j, threadIndex)));
	// }

	// printf("\n < -- simplex %u", threadIndex);

	for (int i = 0; i < dim + 1; i++) {
		for (int j = 0; j < dim; j++) {
			*(Pf.cell(j, i)) = *x0.cell(j, threadIndex);
		}
	}

	for (int i = 0; i < dim; i++) {
		*(Pf.cell(i + 1, i)) = *(Pf.cell(i + 1, i)) + lambda[i];
	}

	for (int i = 0; i < dim + 1; i++) {
		// for (int j=0; j < dim + 1; j++) {
			// printf("PF: %f\n", __half2float(*Pf.cell(i, j)));
		// }
		printf("thread: %u, PF: %u\n", threadIndex, i);
		*Pf.cell(dim, i) = PF(Pf.row(i), yaux, epsin, &localState, hsV);
	}

	k = 0;
	
	Pf = sortm(Pf, dim);
	
	tol = CUDART_ONE_FP16;

	for (k = 0; k < n; k++)
	{
		printf("thread: %u, k: %u\n", threadIndex, k);

		assert(!isnan(__half2float(tol)));
		if (tol < epsilon) {
			break;
		}
		else
		{
			for (int j = 0; j < dim; j++) {
				G[j] = CUDART_ZERO_FP16;
			}

			// for(int i=0;i<dim + 1;i++) {
			// 	for(int j=0;j<dim + 1;j++) {
			// 		printf("%f ", __half2float(*Pf.cell(j, i)));
			// 	}
			// 	printf("\n");
			// }

			// printf("PF\n");

			for (int i = 0; i < dim + 1; i++)
			{
				sumvect(G, Pf.row(i), dim, w); // G is empty, so w gets filled with Pf.row(i)
				initv(G, w, dim);
			}

			// for(int i=0;i<dim + 1;i++) {
			// 	printf("%f ", __half2float(Ptry[i]));
			// }
			// printf(" ptry 1 \n");

			prodcv(G, __float2half(1.0 / (dim + 1)), dim, w);
			initv(G, w, dim); // Copy w to G
			prodcv(Pf.row(dim), __float2half(-1.0), dim, w);
			sumvect(G, w, dim, vec);
			prodcv(vec, __float2half(2.0 * (dim + 1) / dim), dim, w);
			sumvect(Pf.row(dim), w, dim, Ptry);
	
			// for(int i=0;i<dim + 1;i++) {
			// 	printf("%f ", __half2float(Ptry[i]));
			// }
			// printf(" ptry 2 \n");

			ftry = PF(Ptry, yaux, epsin, &localState, hsV);
			if (ftry < *Pf.cell(0, dim))
			{
				prodcv(vec, __float2half(1.0 * (dim + 1) / dim), dim, w);
				sumvect(Ptry, w, dim, Ptry2);
				ftry2 = PF(Ptry2, yaux, epsin, &localState, hsV);
				if (ftry2 < ftry)
				{
					for (int j = 0; j < dim; j++)
						*Pf.cell(dim, j) = Ptry2[j];
					*Pf.cell(dim, dim) = ftry2;
				}
				else
				{
					for (int j = 0; j < dim; j++)
						*Pf.cell(dim, j) = Ptry[j];
					*Pf.cell(dim, dim) = ftry;
				}
			}
			else
			{
				if (ftry > *Pf.cell(dim - 1, dim))
				{
					prodcv(vec, __float2half(0.5 * (dim + 1) / dim), dim, w);
					sumvect(Pf.row(dim), w, dim, Ptry);
					ftry = PF(Ptry, yaux, epsin, &localState, hsV);
					if (ftry > *Pf.cell(dim, dim))
					{
						for (int j = 1; j < dim + 1; j++)
						{
							sumvect(Pf.row(0), Pf.row(j), dim, w);
							prodcv(w, 0.5, dim, z);
							for (int j1 = 0; j1 < dim; j1++) {
								*Pf.cell(j, j1) = z[j1];
							}
							*Pf.cell(j, dim) = PF(Pf.row(j), yaux, epsin, &localState, hsV);
							// printf("4 %f", __half2float(*Pf.cell(j, dim)));
						}
					}
					else
					{
						for (int j = 0; j < dim; j++) {
							// printf("\n2 Ptry[j]%f", __half2float(Ptry[j]));
							*Pf.cell(dim, j) = Ptry[j];
						}
						*Pf.cell(dim, dim) = ftry;
					}
				}
				else
				{
					for (int j = 0; j < dim; j++) {
						// printf("\n1 Ptry[j]%f", __half2float(Ptry[j]));
						*Pf.cell(dim, j) = Ptry[j];
					}
					*Pf.cell(dim, dim) = ftry;
				}
			}
			Pf = sortm(Pf, dim);
		}

		/*  tol=2*(Pf[dim][dim]-Pf[0][dim])/(fabs(Pf[dim][dim])+fabs(Pf[0][dim])+EPS1); */
		tol = __double2half(0.0);

		for (int j = 0; j < dim; j++) {
			// tol += __hmul(2.0,  __hdiv( __habs(__hsub(*Pf.cell(dim, j), *Pf.cell(0, j))) , __hadd(__hadd(__habs(*Pf.cell(dim, j)), __habs(*Pf.cell(0, j))), EPS1) ));
			tol += __hmul(2.0, (__habs(*Pf.cell(dim, j) - *Pf.cell(0, j)) / (__habs(*Pf.cell(dim, j)) + __habs(*Pf.cell(0, j)) + EPS1)));

			// printf("4 first first part nlminvectlsimplex %f\n", __half2float(*Pf.cell(dim, j)));
			// printf("4 first second part nlminvectlsimplex %f\n", __half2float(*Pf.cell(0, j)));

			// printf("4 first part nlminvectlsimplex %f\n", __half2float(__hsub(*Pf.cell(dim, j), *Pf.cell(0, j))));

			// printf("4 second part  nlminvectlsimplex %f\n", __half2float(__hadd(__habs(*Pf.cell(dim, j)), __habs(*Pf.cell(0, j))), EPS1));

			// printf("4 tol nlminvectlsimplex %f\n", __half2float(tol));
		}
	}

	cudaFree(G);
	cudaFree(z);
	cudaFree(Ptry);
	cudaFree(Ptry2);
	cudaFree(w);
	cudaFree(vec);

	if (k < (2 * n - 1))
	{
		/*	printf("NO CONVERGENCE IN %d ITERATIONS\n",n);
			Pf[0][dim]=BIG;*/
	}
}

int main(void)
{
	int devID = 0;
	cudaDeviceProp props;

	// This will pick the best possible CUDA capable device
	cudaSetDevice(devID);

	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);

	// Get GPU information
	// checkCudaErrors(cudaGetDevice(&devID));
	// checkCudaErrors(cudaGetDeviceProperties(&props, devID));
	printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name,
		   props.major, props.minor);


	int i, j, j1, k, n, t, cont, jcont, trials, itno;
	fpxx ftry, ftry1, ftry2, tol, a, sigmad0, val, eps1, sig, minval;
	fpxx *tetaCons;
	fpxx **Pf, **test, **StdAux;

	DMatrix rts, tetainit;

	n = 250;

	FILE *fp; // File pointer
	char s[30], outputFile[100];

	trials = 400; // Number of trials (20)
	itno = 1;	 // Number of iterations

	float* muaux = (float*)malloc(pdim * sizeof(float)); // Vector for storing the auxiliary variables.
	tetaCons = (fpxx *)malloc(pdim * sizeof(fpxx));		 // Vector for storing the constraint variables.
	tetainit = allocmat(pdim, (itno * trials));				 // Matrix for storing the initial constraint variables.

	DMatrix tetainitHost = {
		(fpxx*)malloc(itno * trials * pdim * sizeof(fpxx)),
		pdim,
		itno * trials
	};
	
	DMatrix rtsHost = {
		(fpxx*)malloc(itno * l * sizeof(fpxx)),
		itno,
		l
	};

	rts = allocmat(itno, l);   // Matrix for storing the time series. 1 by 24050
	thrust::device_vector<fpxx> pt(l); // Vector for storing the time series.

	thrust::device_vector<fpxx> hsV(B * trials);

	muaux[0] = -0.190749834;
	muaux[1] = 0.977385697;
	muaux[2] = 0.215634224;
	muaux[3] = 0.015011035;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1); // uniform distribution between 0 and 1

	for (i = 0; i < (itno * trials); i++) // Loops over the trials times itno.
	{
		for (j = 0; j < pdim; j++) { // pdim=4, trials=20, and itno=1
			// For each trial, fill a vector with "muaux[j] * (0.99 + 0.02 * uni())" (uses ziggurat)
			float value = muaux[j] * (0.99 + 0.02 * dis(gen));
			assert(!isnan(value));
			printf("%f ", value);
			*tetainitHost.cell(j, i) = __float2half(value);
		}
		printf("\n");
	}

	const char* filename = "GSPC19280104-20230929.txt";
	// fp = fopen(filename, "r");

	// if (fp == NULL)
	// {
	// 	perror("Failed");
	// 	return 1;
	// }

	// j = 0;

	// // Stores GSPC19280104-20230929.txt data in rts.
	// while (fgets(s, 30, fp) != NULL) // Reads the data from the file line by line. Stops when the end of the file is reached.
	// {
	// 	if (j >= 0)	{			 // This is not needed.
	// 		*rtsHost.cell(0, j) = atof(s); // Fill the first column of rts with the data of the line converted to float.
	// 		// printf("%f\n", *rtsHost.cell(0, j));
	// 	}
	// 	j++; // Increments the rows counter.
	// }
	// fclose(fp);

	// sprintf(outputFile, "SV-T%dN%d_out.txt", l, B); // Generates the output files with enough lines

	// fp = fopen(outputFile, "a");
	// fprintf(fp, "%s	%s	%s	%s	%s\n", "a", "b", "sigma", "mu", "val"); // Sets "a b sigma mu val" in top row of the file
	// fclose(fp);
	// printf("Created output file.\n");

	// lambdaHost[0] = -0.5;
	// lambdaHost[1] = -0.02;
	// lambdaHost[2] = 0.2;
	// lambdaHost[3] = 0.01;

	// cudaMemcpy(lambda, lambdaHost, pdim, cudaMemcpyHostToDevice);

	thrust::device_vector<fpxx> lambdaDevice(pdim);
	lambdaDevice[0] = -0.5;
	lambdaDevice[1] = -0.02;
	lambdaDevice[2] = 0.2;
	lambdaDevice[3] = 0.01;

	minval = BIG; // Big heeft een waarde van 1.0e30
	for (i = 0; i < pdim; i++) {
		tetaCons[i] = *tetainitHost.cell(i, 0); // tetaCons is een vector van 4, tetainit is een matrix van 4x20
	}

	cudaMemcpy(tetainit.ptr, tetainitHost.ptr, tetainitHost.w * tetainitHost.h * sizeof(fpxx), cudaMemcpyHostToDevice);

	fpxx* rtsRow0Device;
	cudaMalloc(&rtsRow0Device, sizeof(fpxx) * rts.w);

	cudaMemcpy(rtsRow0Device, rts.row(0), rts.w * sizeof(fpxx), cudaMemcpyHostToDevice);

	// Main loop

	curandState *states;
	cudaMalloc(&states, sizeof(curandState) * trials);

	DMatrix* pfOut;
	cudaMalloc(&pfOut, sizeof(DMatrix) * trials);

	printf("Starting main function!\n");
	// nlminLvectSimplex<<<1, trials >>>(tetainit, 2000, lambda, rts.row(0), pt, Weightmatrix, Weightmatrix, EPS1, pdim, states, hsV);
	nlminLvectSimplex<<<trials / 100, 100>>>(
		tetainit, 
		2000, 
		thrust::raw_pointer_cast(lambdaDevice.data()), 
		rts.row(0), 
		thrust::raw_pointer_cast(pt.data()), 
		EPS1, 
		pdim, 
		states, 
		thrust::raw_pointer_cast(hsV.data()), 
		pfOut
	);

	cudaDeviceSynchronize();

	DMatrix* pfOutHost = (DMatrix*) malloc(trials * sizeof(DMatrix));
	cudaMemcpy(pfOutHost, pfOut, trials, cudaMemcpyDeviceToHost);

	printf("done with simplex");

	for (jcont = 0; jcont < trials; jcont++) 
	{
		DMatrix pf = pfOutHost[jcont];
		
		fpxx* halfs = (fpxx*) malloc(sizeof(fpxx) * pf.w * pf.h);
		cudaMemcpy(halfs, pf.ptr, pf.w * pf.h * sizeof(fpxx), cudaMemcpyDeviceToHost);

		// pf.free();
		pf.ptr = halfs;

		*pf.cell(0, 2) = __float2half(fabs(__half2float(*pf.cell(0, 2))));

		//max val of fp16, or so
		fpxx minval = __float2half(66504.0);

		if (*pf.cell(0, pdim) < minval)
		{
			for (i = 0; i < pdim; i++) {
				tetaCons[i] = *pf.cell(0, i);
			}
			minval = *pf.cell(0, pdim);
		}

		fp = fopen(outputFile, "a");
		for (i = 0; i < pdim; i++) {
			fprintf(fp, "%.12f	", *pf.cell(0, i));
		}
		fprintf(fp, "%.16f\n", *pf.cell(0, pdim));
		fclose(fp);

		/*	printf("jcont=%d\n",jcont);
			writemat("t",pdim+1,pdim+1,test);*/

		// free(pf.ptr);
	}

	// This part of the code frees the memory allocated for the matrices and vectors.
	free(muaux);
	rts.free();
	free(tetaCons);
	tetainit.free();

	exit(0);
}