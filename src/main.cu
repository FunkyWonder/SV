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
	DMatrix x0, int n, double *lambda, double *yaux, double *epsin, double epsilon, int dim, curandState *globalRands, double *hsV)
{
	int threadIndex = threadIdx.x + (blockDim.x + threadIdx.y);
	curand_init(0, threadIndex, 0, &globalRands[threadIndex]);

	curandState localState = globalRands[threadIndex];

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

	for (i = 0; i < dim + 1; i++) {
		for (j = 0; j < dim; j++) {
			*(Pf.cell(i, j)) = *x0.cell(threadIndex, j);
		}
	}
  
	for (i = 0; i < dim; i++) {
		*(Pf.cell(i + 1, i)) = *(Pf.cell(i + 1, i)) + lambda[i];
	}


	for (i = 0; i < dim + 1; i++) {
		*Pf.cell(i, dim) = PF(Pf.cell(i, 0), yaux, epsin, &localState, hsV);
	}

	k = 0;

	
	Pf = sortm(Pf, dim);

	tol = 1.0;

	for (k = 0; k < n; k++)
	{
		printf("%f", tol);
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
			ftry = PF(Ptry, yaux, epsin, &localState, hsV);
			if (ftry < *Pf.cell(0, dim))
			{
				prodcv(vec, 1.0 * (dim + 1) / dim, dim, w);
				sumvect(Ptry, w, dim, Ptry2);
				ftry2 = PF(Ptry2, yaux, epsin, &localState, hsV);
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
					ftry = PF(Ptry, yaux, epsin, &localState, hsV);
					if (ftry > *Pf.cell(dim, dim))
					{
						for (j = 1; j < dim + 1; j++)
						{
							sumvect(Pf.row(0), Pf.row(j), dim, w);
							prodcv(w, 0.5, dim, z);
							for (j1 = 0; j1 < dim; j1++)
								*Pf.cell(j, j1) = z[j1];
							*Pf.cell(j, dim) = PF(Pf.row(j), yaux, epsin, &localState, hsV);
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
	double ftry, ftry1, ftry2, tol, a, sigmad0, val, eps1, sig, minval;
	double *tetaCons;
	double **Pf, **test, **StdAux;

	DMatrix rts, tetainit;

	n = 250;

	FILE *fp; // File pointer
	char s[30], outputFile[100];

	trials = 1; // Number of trials
	itno = 1;	 // Number of iterations

	double *muaux = (double *)malloc(pdim * sizeof(double)); // Vector for storing the auxiliary variables.
	tetaCons = (double *)malloc(pdim * sizeof(double));		 // Vector for storing the constraint variables.
	tetainit = allocmat((itno * trials), pdim);				 // Matrix for storing the initial constraint variables.

	DMatrix tetainitHost = {
		(double *)malloc(itno * trials * pdim * sizeof(double)),
		itno * trials,
		pdim
	};
	
	DMatrix rtsHost = {
		(double *)malloc(itno * l * sizeof(double)),
		itno,
		l
	};

	rts = allocmat(itno, l);   // Matrix for storing the time series. 1 by 24050
	thrust::device_vector<double> pt(l); // Vector for storing the time series.

	thrust::device_vector<double> hsV(B);

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

	thrust::device_vector<double> lambdaDevice(pdim);
	lambdaDevice[0] = -0.5;
	lambdaDevice[1] = -0.02;
	lambdaDevice[2] = 0.2;
	lambdaDevice[3] = 0.01;

	minval = BIG; // Big heeft een waarde van 1.0e30
	for (i = 0; i < pdim; i++) {
		tetaCons[i] = *tetainitHost.cell(0, i); // tetaCons is een vector van 4, tetainit is een matrix van 4x20
	}

	cudaMemcpy(tetainit.ptr, tetainitHost.ptr, tetainitHost.w * tetainitHost.h * sizeof(double), cudaMemcpyHostToDevice);

	double* rtsRow0Device;
	cudaMalloc(&rtsRow0Device, sizeof(double) * rts.w);

	cudaMemcpy(rtsRow0Device, rts.row(0), rts.w * sizeof(double), cudaMemcpyHostToDevice);

	// Main loop

	curandState *states;
	cudaMalloc(&states, sizeof(curandState) * trials);

	printf("Starting main function!\n");
	// nlminLvectSimplex<<<1, trials >>>(tetainit, 2000, lambda, rts.row(0), pt, Weightmatrix, Weightmatrix, EPS1, pdim, states, hsV);
	nlminLvectSimplex<<<1, trials>>>(tetainit, 2000, thrust::raw_pointer_cast(lambdaDevice.data()), rts.row(0), thrust::raw_pointer_cast(pt.data()), EPS1, pdim, states, thrust::raw_pointer_cast(hsV.data()));

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

	exit(0);
}
