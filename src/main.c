#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <math.h>
#include <malloc.h>

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

typedef unsigned int uint32;
typedef signed int int32;

static uint32 jz, jsr = 123456789;

int lags, invcont, kbar, l, contS, seed, B;
double D1p, pd1, logsqrt2pi, lbound, phiconst, sqrt2, csqtr, m0true, gammakbtrue, btrue, sigmadtrue, alphaaux, sqrt2dPi;
double *muaux, *hsV, *driftmut, *monthmut, *shortmut, *ma50t, *ma250t;
double **Weightmatrix;

double *alocvec(int n)
{
	double *z;

	z = (double *)malloc(sizeof(double) * n);
	if (z == NULL)
	{
		printf("out of memory\n");
		exit(1);
	}

	return z;
}

double **alocmat(int m, int n)
{
	double **z;
	int i;

	z = (double **)malloc(sizeof(double *) * m);
	if (z == NULL)
	{
		printf("out of memory\n");
		exit(1);
	}
	for (i = 0; i < m; i++)
	{
		z[i] = (double *)malloc(sizeof(double) * n);
		if (z[i] == NULL)
		{
			printf("out of memory\n");
			exit(1);
		}
	}
	return z;
}

int freemat(double **P, int m)
{
	int i;

	for (i = 0; i < m; i++)
		free(P[i]);
	free(P);
	return 1;
}

int *alocintvec(int n)
{
	int *z;
	z = (int *)malloc(sizeof(int) * n);
	if (z == NULL)
	{
		printf("out of memory\n");
		exit(1);
	}

	return z;
}

int initm(double **M1, double **M2, int m, int n)
{
	double **P, **R, *Pv, *Rv;
	int i, j;

	P = M1;
	R = M2;

	for (i = 0; i < m; i++)
	{
		Pv = *P;
		Rv = *R;
		for (j = 0; j < n; j++)
		{
			*Pv = *Rv;
			Pv++;
			Rv++;
		}
		P++;
		R++;
	}

	return 1;
}

int initv(double *U, double *V, int n)
{
	double *T, *W;
	int i;

	T = U;
	W = V;

	for (i = 0; i < n; i++)
	{
		*T = *W;
		T++;
		W++;
	}

	return 1;
}

int initm0(double **M, int m, int n)
{
	double **P, *Pv;
	int i, j;

	P = M;

	for (i = 0; i < m; i++)
	{
		Pv = *P;
		for (j = 0; j < n; j++)
		{
			*Pv = 0.0;
			Pv++;
		}
		P++;
	}

	return 1;
}

double sum(double *yn, int i0, int it)
{
	int i;
	double s;

	s = 0;
	for (i = i0; i <= it; i++)
		s = s + yn[i];
	return s;
}

int prodcv(double *y1, double c, int n, double *res)
{
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

int sumvect(double *y1, double *y2, int n, double *res)
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

double prodscal(double *yn, double *ym, int i0, int it)
{
	int i;
	double s;

	s = 0.0;
	for (i = i0; i <= it; i++)
		s += yn[i] * ym[i];
	return s;
}

int trans(double **P, int m, int n, double **res)
{
	double *T;
	int i, j;

	for (i = 0; i < m; i++)
	{
		T = *P;
		for (j = 0; j < n; j++)
		{
			res[j][i] = (*T);
			T++;
		}
		P++;
	}

	return 1;
}

int prodmat(double **P1, double **P2, int m, int n, int p, double **res)
{
	double **P, **M1, **M2, *M11, *Pv;
	int i, j, k;

	P = res;
	M1 = P1;

	for (i = 0; i < m; i++)
	{

		Pv = *P;
		for (j = 0; j < p; j++)
		{
			(*Pv) = 0;
			M11 = (*M1);
			M2 = P2;
			for (k = 0; k < n; k++)
			{
				(*Pv) += (*M11) * ((*M2)[j]);
				M11++;
				M2++;
			}
			Pv++;
		}
		M1++;
		P++;
	}

	return 1;
}

int writemat(char *s, int m, int n, double **P)
{
	int i, j;
	double **M;
	double *x;

	M = P;

	for (i = 0; i < m; i++)
	{
		x = *M;
		for (j = 0; j < n; j++)
		{
			printf("%s[%d,%d]=%.10f ", s, i + 1, j + 1, *x);
			x++;
		}
		printf("\n");
		M++;
	}
	return 1;
}

int inv(double **A, int n, double **P)
{
	int *indxc, *indxr, *ipiv;
	double *aux;
	int i, icol, irow, j, k, l, ll;
	double big, dum, pivinv, aux1;

	invcont = 0;

	// The integer arrays ipiv, indxr, and indxc are used for bookkeeping on the pivoting.

	indxc = alocintvec(n); // for the index start from 1 in the book
	indxr = alocintvec(n);
	ipiv = alocintvec(n);

	initm(P, A, n, n);

	for (j = 0; j < n; j++)
		ipiv[j] = 0;

	for (i = 0; i < n; i++) // the main loop over the columns to bereduced.
	{
		big = 0.0;
		for (j = 0; j < n; j++) // the outer loop of the search for a pivot element.
		{
			if (ipiv[j] != 1)
				for (k = 0; k < n; k++)
				{
					if (ipiv[k] == 0)
					{
						if (fabs(P[j][k]) >= big)
						{
							big = fabs(P[j][k]);
							irow = j;
							icol = k;
						}
					}
					else if (ipiv[k] > 1)
					{
						/*printf("gaussj: Singular Matrix-1\n");*/
						invcont = 1;
					}
				}
		}
		(ipiv[icol])++;
		/*We now have the pivot element, so we interchange rows, if needed, to put the pivot
		element on the diagonal. The columns are not physically interchanged, only relabeled:
		indxc[i], the column of the ith pivot element, is the ith column that is reduced, while
		indxr[i] is the row in which that pivot element was originally located.
		If indxr[i] != indxc[i] there is an implied column interchange. With this form of
		bookkeeping, the solution b's will end up in the correct order, and the inverse matrix
		will be scrambled by columns. */

		if (irow != icol)
		{
			aux = P[irow];
			P[irow] = P[icol];
			P[icol] = aux;
		}
		indxr[i] = irow; // We are now ready to divide the pivot row by the
		indxc[i] = icol; // pivot element, located at irow and icol.
		if (P[icol][icol] == 0.0)
		{
			/*printf("gaussj: Singular Matrix-2\n");*/
			invcont = 1;
		}
		pivinv = 1.0 / P[icol][icol];
		P[icol][icol] = 1.0;
		for (l = 0; l < n; l++)
			P[icol][l] *= pivinv;
		for (ll = 0; ll < n; ll++) // Next, we reduce the rows...
		{
			if (ll != icol) //...except for the pivot one, of course.
			{
				dum = P[ll][icol];
				P[ll][icol] = 0.0;
				for (l = 0; l < n; l++)
					P[ll][l] -= P[icol][l] * dum;
			}
		}
	} // main loop over columns of the reduction.

	/*
	It only remains to unscramble the solution in view of the column interchanges.
	We do this by interchanging pairs of columns in the reverse order that the
	permutation was built up. */

	for (l = n - 1; l >= 0; l--)

		if (indxr[l] != indxc[l])
			for (k = 0; k < n; k++)
			{
				aux1 = P[k][indxr[l]];
				P[k][indxr[l]] = P[k][indxc[l]];
				P[k][indxc[l]] = aux1;
			}
	free(ipiv);
	free(indxr);
	free(indxc);

	return 1;
}

double **sortm(double **P, int ds)
{
	double *z;
	double Min, Max, Max2;
	int i, imin, imax, imax2;

	Min = P[0][ds];
	imin = 0;

	for (i = 1; i < ds + 1; i++)
		if (P[i][ds] < Min)
		{
			Min = P[i][ds];
			imin = i;
		}

	if (imin != 0)
	{
		z = P[imin];
		P[imin] = P[0];
		P[0] = z;
	}

	Max = P[ds][ds];
	imax = ds;

	for (i = 1; i < ds; i++)
		if (P[i][ds] > Max)
		{
			Max = P[i][ds];
			imax = i;
		}

	if (imax != ds)
	{
		z = P[imax];
		P[imax] = P[ds];
		P[ds] = z;
	}

	Max2 = P[ds - 1][ds];
	imax2 = ds - 1;

	for (i = 1; i < ds - 1; i++)
		if (P[i][ds] > Max2)
		{
			Max2 = P[i][ds];
			imax2 = i;
		}
	if (imax2 != ds - 1)
	{
		z = P[imax2];
		P[imax2] = P[ds - 1];
		P[ds - 1] = z;
	}

	return P;
}

double **nlminLvectSimplex(double (*func)(double *, double *, double *, double **, double **),
						   double *x0, int n, double *lambda, double *yaux, double *epsin, double **At, double **yni, double epsilon, int dim)
{

	double **Pf;

	double *G, *z, *Ptry, *Ptry2, *w, *vec;

	double ftry, ftry2, tol;

	int i, j, j1, k;

	G = alocvec(dim);
	z = alocvec(dim);
	Pf = alocmat(dim + 1, dim + 1);
	Ptry = alocvec(dim + 1);
	Ptry2 = alocvec(dim + 1);
	w = alocvec(dim);
	vec = alocvec(dim);

	for (i = 0; i < dim + 1; i++)
		for (j = 0; j < dim; j++)
			Pf[i][j] = x0[j];

	for (i = 0; i < dim; i++)
		Pf[i + 1][i] = Pf[i + 1][i] + lambda[i];

	for (i = 0; i < dim + 1; i++)
		Pf[i][dim] = func(Pf[i], yaux, epsin, At, yni);

	k = 0;

	Pf = sortm(Pf, dim);

	tol = 1.0;

	for (k = 0; k < n; k++)
	{
		if (tol < epsilon)
		{
			/*		printf("iter=%d\n",k);*/
			k = 2 * n;
		}
		else
		{
			for (j = 0; j < dim; j++)
				G[j] = 0.0;
			for (i = 0; i < dim + 1; i++)
			{
				sumvect(G, Pf[i], dim, w);
				initv(G, w, dim);
			}
			prodcv(G, 1.0 / (dim + 1), dim, w);
			initv(G, w, dim);
			prodcv(Pf[dim], -1.0, dim, w);
			sumvect(G, w, dim, vec);
			prodcv(vec, 2.0 * (dim + 1) / dim, dim, w);
			sumvect(Pf[dim], w, dim, Ptry);
			ftry = func(Ptry, yaux, epsin, At, yni);
			if (ftry < Pf[0][dim])
			{
				prodcv(vec, 1.0 * (dim + 1) / dim, dim, w);
				sumvect(Ptry, w, dim, Ptry2);
				ftry2 = func(Ptry2, yaux, epsin, At, yni);
				if (ftry2 < ftry)
				{
					for (j = 0; j < dim; j++)
						Pf[dim][j] = Ptry2[j];
					Pf[dim][dim] = ftry2;
				}
				else
				{
					for (j = 0; j < dim; j++)
						Pf[dim][j] = Ptry[j];
					Pf[dim][dim] = ftry;
				}
			}
			else
			{
				if (ftry > Pf[dim - 1][dim])
				{
					prodcv(vec, 0.5 * (dim + 1) / dim, dim, w);
					sumvect(Pf[dim], w, dim, Ptry);
					ftry = func(Ptry, yaux, epsin, At, yni);
					if (ftry > Pf[dim][dim])
					{
						for (j = 1; j < dim + 1; j++)
						{
							sumvect(Pf[0], Pf[j], dim, w);
							prodcv(w, 0.5, dim, z);
							for (j1 = 0; j1 < dim; j1++)
								Pf[j][j1] = z[j1];
							Pf[j][dim] = func(Pf[j], yaux, epsin, At, yni);
						}
					}
					else
					{
						for (j = 0; j < dim; j++)
							Pf[dim][j] = Ptry[j];
						Pf[dim][dim] = ftry;
					}
				}
				else
				{
					for (j = 0; j < dim; j++)
						Pf[dim][j] = Ptry[j];
					Pf[dim][dim] = ftry;
				}
			}
			Pf = sortm(Pf, dim);
		}
		/*  tol=2*(Pf[dim][dim]-Pf[0][dim])/(fabs(Pf[dim][dim])+fabs(Pf[0][dim])+EPS1); */
		tol = 0.0;
		for (j = 0; j < dim; j++)
			tol += 2.0 * fabs(Pf[dim][j] - Pf[0][j]) / (fabs(Pf[dim][j]) + fabs(Pf[0][j]) + EPS1);
	}

	if (k < (2 * n - 1))
	{
		/*	printf("NO CONVERGENCE IN %d ITERATIONS\n",n);
			Pf[0][dim]=BIG;*/
	}

	free(G);
	free(z);
	free(Ptry);
	free(Ptry2);
	free(w);
	free(vec);

	return Pf;
}

void lnsrch(int n, double *xold, double fold, double *g, double *p, double *x,
			double *f, double stpmax, int *check, double (*func)(double *, double *, double *, double **, double **),
			double *y1, double *y2, double **Ai, double **z)
/*Given an n-dimensional point xold[0..n-1], the value of the function and gradient there, fold
and g[0..n-1], and a direction p[0..n-1], finds a new point x[0..n] along the direction p from
xold where the function func has decreased sufficiently. The new function value is returned
in f. stpmax is an input quantity that limits the length of the steps so that you do not try to
evaluate the function in regions where it is undefined or subject to overflow. p is usually the
Newton direction. The output quantity check is false (0) on a normal exit. It is true (1) when
x is too close to xold. In a minimization algorithm, this usually signals convergence and can
be ignored. However, in a zero-finding algorithm the calling program should check whether the
convergence is spurious. Some difficult problems may require double precision in this routine.*/
{
	int i;
	double a, alam, alam2, alamin, b, disc, f2, rhs1, rhs2, slope, sum, temp, test, tmplam;

	*check = 0;
	for (sum = 0.0, i = 0; i < n; i++)
		sum += p[i] * p[i];
	sum = sqrt(sum);
	if (sum > stpmax)
		for (i = 0; i < n; i++)
			p[i] *= stpmax / sum; /*Scale if attempted step is too big.*/

	for (slope = 0.0, i = 0; i < n; i++)
		slope += g[i] * p[i];

	if (slope >= 0.0)
		printf("Roundoff problem in lnsrch.");
	test = 0.0; /*Compute lambda_min.*/
	for (i = 0; i < n; i++)
	{
		temp = fabs(p[i]) / (fabs(xold[i]) > 1.0 ? fabs(xold[i]) : 1.0);
		if (temp > test)
			test = temp;
	}
	alamin = TOLX / test;

	alam = 1.0; /*Always try full Newton step first.*/
	for (;;)
	{ /*Start of iteration loop.*/
		for (i = 0; i < n; i++)
			x[i] = xold[i] + alam * p[i];
		*f = (*func)(x, y1, y2, Ai, z);
		if (alam < alamin)
		{ /*Convergence on delta_x. For zero finding,the calling program should
			 verify the convergence. */
			for (i = 0; i < n; i++)
				x[i] = xold[i];
			*check = 1;
			return;
		}
		else if (*f <= fold + ALF * alam * slope)
		{
			return; /*Sufficient function decrease.*/
		}

		else
		{ /*Backtrack.*/
			if (alam == 1.0)
				tmplam = -slope / (2.0 * (*f - fold - slope)); /*First time.*/
			else
			{ /*Subsequent backtracks.*/
				rhs1 = *f - fold - alam * slope;
				rhs2 = f2 - fold - alam2 * slope;
				a = (rhs1 / (alam * alam) - rhs2 / (alam2 * alam2)) / (alam - alam2);
				b = (-alam2 * rhs1 / (alam * alam) + alam * rhs2 / (alam2 * alam2)) / (alam - alam2);
				if (a == 0.0)
					tmplam = -slope / (2.0 * b);
				else
				{
					disc = b * b - 3.0 * a * slope;
					if (disc < 0.0)
						tmplam = 0.5 * alam;
					else if (b <= 0.0)
						tmplam = (-b + sqrt(disc)) / (3.0 * a);
					else
						tmplam = -slope / (b + sqrt(disc));
				}
				if (tmplam > 0.5 * alam)
					tmplam = 0.5 * alam; /* lambda<=0.5*lamdbda_1.*/
			}
		}
		alam2 = alam;
		f2 = *f;
		alam = (tmplam > (0.1 * alam) ? tmplam : (0.1 * alam)); /* lambda>=0.1*lamdbda_1.*/

	} /*Try again.*/
}

void dfpmin(double *p, int n, double gtol, double *fret,
			double (*func)(double *, double *, double *, double **, double **), void (*dfunc)(double *, double *, double *, double *, double **, double **),
			double *y, double *z, double **Ai, double **yni)
/*Given a starting point p[0..n-1] that is a vector of length n, the Broyden-Fletcher-Goldfarb-
Shanno variant of Davidon-Fletcher-Powell minimization is performed on a function func, using
its gradient as calculated by a routine dfunc. The convergence requirement on zeroing the
gradient is input as gtol. Returned quantities are p[0..n-1] (the location of the minimum),
iter (the number of iterations that were performed), and fret (the minimum value of the
function). The routine lnsrch is called to perform approximate line minimizations.*/
{
	void lnsrch(int n, double *xold, double fold, double *g, double *p, double *x,
				double *f, double stpmax, int *check, double (*func)(double *, double *, double *, double **, double **),
				double *y, double *z, double **Ai, double **yni);

	int check, i, its, j;
	double den, fac, fad, fae, fp, stpmax, sum = 0.0, sumdg, sumxi, temp, test;
	double *dg, *g, *hdg, **hessin, *pnew, *xi;

	dg = alocvec(n);
	g = alocvec(n);
	hdg = alocvec(n);
	hessin = alocmat(n, n);
	pnew = alocvec(n);
	xi = alocvec(n);
	fp = (*func)(p, y, z, Ai, yni);
	/*Calculate starting function value and gradient, */
	(*dfunc)(p, g, y, z, Ai, yni);
	for (i = 0; i < n; i++)
	{
		/*and initialize the inverse Hessian to the unit matrix.*/
		for (j = 0; j < n; j++)
			hessin[i][j] = 0.0;
		hessin[i][i] = 1.0;
		xi[i] = -g[i]; /*Initial line direction.*/
		sum += p[i] * p[i];
	}
	stpmax = STPMX * (sqrt(sum) > (double)n ? sqrt(sum) : (double)n);
	for (its = 0; its < ITMAX; its++)
	{ /*Main loop over the iterations.*/
		lnsrch(n, p, fp, g, xi, pnew, fret, stpmax, &check, (*func), y, z, Ai, yni);
		/*The new function evaluation occurs in lnsrch; save the function
		  value in fp for the next line search. It is usually safe to
		  ignore the value of check.*/

		fp = *fret;
		for (i = 0; i < n; i++)
		{
			xi[i] = pnew[i] - p[i]; /*Update the line direction,*/
			p[i] = pnew[i];			/*and the current point.*/
		}
		test = 0.0; /*Test for convergence on �.*/
		for (i = 0; i < n; i++)
		{
			temp = fabs(xi[i]) / (fabs(p[i]) > 1.0 ? fabs(p[i]) : 1.0);
			if (temp > test)
				test = temp;
		}
		if (test < TOLX)
		{
			FREEALL
			return;
		}
		for (i = 0; i < n; i++)
			dg[i] = g[i];			   /*Save the old gradient, */
		(*dfunc)(p, g, y, z, Ai, yni); /*and get the new gradient.*/

		test = 0.0; /*Test for convergence on zero gradient.*/
		den = ((*fret) > 1.0 ? (*fret) : 1.0);
		for (i = 0; i < n; i++)
		{
			temp = fabs(g[i]) * (fabs(p[i]) > 1.0 ? fabs(p[i]) : 1.0) / den;
			if (temp > test)
				test = temp;
		}
		if (test < gtol)
		{
			FREEALL
			return;
		}
		for (i = 0; i < n; i++)
			dg[i] = g[i] - dg[i]; /*Compute difference of gradients,*/
		for (i = 0; i < n; i++)
		{ /*and difference times current matrix.*/
			hdg[i] = 0.0;
			for (j = 0; j < n; j++)
				hdg[i] += hessin[i][j] * dg[j];
		}
		fac = fae = sumdg = sumxi = 0.0;
		/*Calculate dot products for the denominators.*/
		for (i = 0; i < n; i++)
		{
			fac += dg[i] * xi[i];
			fae += dg[i] * hdg[i];
			sumdg += pow((dg[i]), 2.0);
			sumxi += pow((xi[i]), 2.0);
		}
		if (fac > sqrt(EPS * sumdg * sumxi))
		{
			/*Skip update if fac not sufficiently positive. */
			fac = 1.0 / fac;
			fad = 1.0 / fae;
			/* The vector that makes BFGS different from DFP: */
			for (i = 0; i < n; i++)
				dg[i] = fac * xi[i] - fad * hdg[i];
			for (i = 0; i < n; i++)
			{ /*The BFGS updating formula:*/
				for (j = 0; j < n; j++)
				{
					hessin[i][j] += fac * xi[i] * xi[j] - fad * hdg[i] * hdg[j] + fae * dg[i] * dg[j];
					/*hessin[j][i]=hessin[i][j];*/
				}
			}
		}
		for (i = 0; i < n; i++)
		{ /*Now calculate the next direction to go,*/
			xi[i] = 0.0;
			for (j = 0; j < n; j++)
				xi[i] -= hessin[i][j] * g[j];
		}
	} /*and go back for another iteration.*/
	printf("too many iterations in dfpmin");
	FREEALL
}

double dfridr(double (*func)(double, double *, int, int, double *, double *, double **, double **), double x, double *teta, int id, int jd, double h, double *yaux, double *epsin, double **Ai, double **yni)
/*Returns the derivative of a function func at a point x by Ridders method of polynomial
extrapolation. The value h is input as an estimated initial stepsize; it need not be small, but
rather should be an increment in x over which func changes substantially. An estimate of the
error in the derivative is returned as err.*/
{
	int i, j;
	double errt, fac, hh, **a, ans, err;
	if (h == 0.0)
		printf("h must be nonzero in dfridr.");
	a = alocmat(10, 10);
	hh = h;

	a[0][0] = ((*func)(x + hh, teta, id, jd, yaux, epsin, Ai, yni) - (*func)(x - hh, teta, id, jd, yaux, epsin, Ai, yni)) / (2.0 * hh);

	err = pow(10.0, 10);

	for (i = 1; i < 10; i++)
	{
		/*Successive columns in the Neville tableau will go to smaller stepsizes and higher orders of
		extrapolation.*/
		hh /= CON;
		a[0][i] = ((*func)(x + hh, teta, id, jd, yaux, epsin, Ai, yni) - (*func)(x - hh, teta, id, jd, yaux, epsin, Ai, yni)) / (2.0 * hh); /*Try new, smaller stepsize.*/
		fac = CON2;
		for (j = 1; j <= i; j++)
		{ /*Compute extrapolations of various orders, requiring no new function evaluations.*/
			a[j][i] = (a[j - 1][i] * fac - a[j - 1][i - 1]) / (fac - 1.0);
			fac = CON2 * fac;
			errt = (fabs(a[j][i] - a[j - 1][i]) > fabs(a[j][i] - a[j - 1][i - 1]) ? fabs(a[j][i] - a[j - 1][i]) : fabs(a[j][i] - a[j - 1][i - 1]));

			/*The error strategy is to compare each new extrapolation to one order lower, both
			at the present stepsize and the previous one.*/
			if (errt <= err)
			{ /*If error is decreased, save the improved answer.*/
				err = errt;
				ans = a[j][i];
			}
		}
		if (fabs(a[i][i] - a[i - 1][i - 1]) >= SAFE * (err))
			break;
		/*If higher order is worse by a significant factor SAFE, then quit early.*/
	}
	freemat(a, 10);
	return ans;
}

void dichotom(double (*func)(double *, double *, double *, double **, double **), double *z1, double *z2, double **z3, double **z4, double tol, double low, double up, double *x)
{
	int i;
	double x1, x2, x3, val1, val2, val3;

	x[0] = low;
	x1 = low;
	val1 = (*func)(x, z1, z2, z3, z4);
	x[0] = up;
	x2 = up;
	val2 = (*func)(x, z1, z2, z3, z4);
	val3 = 100.0;

	i = 0;
	while ((fabs(val3) > tol) && (i < 200))
	{
		x[0] = (x1 + x2) / 2.0;
		x3 = x[0];
		val3 = (*func)(x, z1, z2, z3, z4);
		if (val3 * val1 > 0.0)
		{
			x1 = x3;
			val1 = val3;
		}
		else
		{
			x2 = x3;
			val2 = val3;
		}
		i += 1;
	}
	/*if (i>199) printf("val3=%.12f\n",val3);*/
}

/* The ziggurat method for RNOR and REXP
Combine the code below with the main program in which you want
normal or exponential variates.   Then use of RNOR in any expression
will provide a standard normal variate with mean zero, variance 1,
while use of REXP in any expression will provide an exponential variate
with density exp(-x),x>0.
Before using RNOR or REXP in your main, insert a command such as
zigset(86947731 );
with your own choice of seed value>0, rather than 86947731.
(If you do not invoke zigset(...) you will get all zeros for RNOR and REXP.)
For details of the method, see Marsaglia and Tsang, "The ziggurat method
for generating random variables", Journ. Statistical Software.
*
*
* Fixed for 64-bit GCC March 2009 Phil Karn
* $Id$
*/

static inline uint32 shr3(void)
{
	jz = jsr;
	jsr ^= (jsr << 13);
	jsr ^= (jsr >> 17);
	jsr ^= (jsr << 5);
	return jz + jsr;
}

static inline float uni(void)
{
	return .5 + (int32)shr3() * .2328306e-9;
}

static int32 hz;
static uint32 iz, kn[128], ke[256];
static float wn[128], fn[128], we[256], fe[256];

#define RNOR (hz = shr3(), iz = hz & 127, (fabs(hz) < kn[iz]) ? hz * wn[iz] : nfix())
#define REXP (jz = shr3(), iz = jz & 255, (jz < ke[iz]) ? jz * we[iz] : efix())

/* nfix() generates variates from the residue when rejection in RNOR occurs. */

float nfix(void)
{
	const float r = 3.442620f; /* The start of the right tail */
	static float x, y;
	for (;;)
	{
		x = hz * wn[iz]; /* iz==0, handles the base strip */
		if (iz == 0)
		{
			do
			{
				x = -log(uni()) * 0.2904764;
				y = -log(uni());
			} /* .2904764 is 1/r */
			while (y + y < x * x);
			return (hz > 0) ? r + x : -r - x;
		}
		/* iz>0, handle the wedges of other strips */
		if (fn[iz] + uni() * (fn[iz - 1] - fn[iz]) < exp(-.5 * x * x))
			return x;

		/* initiate, try to exit for(;;) for loop*/
		hz = shr3();
		iz = hz & 127;
		if (fabs(hz) < kn[iz])
			return (hz * wn[iz]);
	}
}

/* efix() generates variates from the residue when rejection in REXP occurs. */
float efix(void)
{
	float x;
	for (;;)
	{
		if (iz == 0)
			return (7.69711 - log(uni())); /* iz==0 */
		x = jz * we[iz];
		if (fe[iz] + uni() * (fe[iz - 1] - fe[iz]) < exp(-x))
			return (x);

		/* initiate, try to exit for(;;) loop */
		jz = shr3();
		iz = (jz & 255);
		if (jz < ke[iz])
			return (jz * we[iz]);
	}
}
/*--------This procedure sets the seed and creates the tables------*/

/* Set up tables for RNOR */
void zigset_nor(uint32 jsrseed)
{
	const double m1 = 2147483648.0;
	double dn = 3.442619855899, tn = dn, vn = 9.91256303526217e-3, q;
	int i;

	jsr ^= jsrseed;

	q = vn / exp(-.5 * dn * dn);
	kn[0] = (dn / q) * m1;
	kn[1] = 0;

	wn[0] = q / m1;
	wn[127] = dn / m1;

	fn[0] = 1.;
	fn[127] = exp(-.5 * dn * dn);

	for (i = 126; i >= 1; i--)
	{
		dn = sqrt(-2. * log(vn / dn + exp(-.5 * dn * dn)));
		kn[i + 1] = (dn / tn) * m1;
		tn = dn;
		fn[i] = exp(-.5 * dn * dn);
		wn[i] = dn / m1;
	}
}

/* Set up tables for REXP */
void zigset_exp(uint32 jsrseed)
{
	const double m2 = 4294967296.;
	double q;
	double de = 7.697117470131487, te = de, ve = 3.949659822581572e-3;
	int i;

	// doing it twice, once from zigset_exp and from zigset_nor, cancels the effect!
	//   jsr^=jsrseed; //

	q = ve / exp(-de);
	ke[0] = (de / q) * m2;
	ke[1] = 0;

	we[0] = q / m2;
	we[255] = de / m2;

	fe[0] = 1.;
	fe[255] = exp(-de);

	for (i = 254; i >= 1; i--)
	{
		de = -log(ve / de + exp(-de));
		ke[i + 1] = (de / te) * m2;
		te = de;
		fe[i] = exp(-de);
		we[i] = de / m2;
	}
}
/* Set up tables */
void zigset(uint32 jsrseed)
{
	zigset_nor(jsrseed);
	zigset_exp(jsrseed);
}

float rnor()
{
	hz = shr3();
	iz = hz & 127;
	if (fabs(hz) < kn[iz])
		return hz * wn[iz];
	else
		return nfix();
}
float rexp()
{
	jz = shr3();
	iz = jz & 255;
	if (jz < ke[iz])
		return jz * we[iz];
	else
		return efix();
}

/********** Generation of bivariate normal random variables ********/

int binorm(double r1, double r2, double rho, double mu1, double mu2, double sigma1, double sigma2, double *rbin)
{

	rbin[0] = mu1 + sigma1 * r1;
	rbin[1] = mu2 + rho * sigma2 * r1 + pow(pow(sigma2, 2.0) * (1.0 - pow(rho, 2.0)), 0.5) * r2;

	return 1;
}

/********* Pdf of univariate normal *************/

double pdf_norm(double x, double mu, double sigma)
{
	double val;

	val = (x - mu) / sigma;
	val = -pow(val, 2.0) / 2;
	val = exp(val) / (sqrt(2 * miPi) * sigma);

	if (fabs(val) > 0.0)
		return val;
	else
		return 0.0;
}

double pdf_Cauchy(double x, double mu, double sigma)
{
	double val, const0;

	const0 = sigma * miPi / 2;

	val = (x - mu) / sigma;
	val = pow(val, 2.0);
	val = pow(1.0 + val, -2.0) / const0;

	return val;
}

/********** Pdf of a bivariate normal ****************/

double pdf_bin(double x, double y, double mux, double muy, double sigmax, double sigmay, double ro)
{
	double val, Pi;

	Pi = 3.141592654;

	val = exp(-(pow((x - mux) / sigmax, 2.0) - 2 * ro * (x - mux) * (y - muy) / (sigmax * sigmay) + pow((y - muy) / sigmay, 2.0)) / (2 * (1.0 - pow(ro, 2.0)))) / (2 * Pi * sigmax * sigmay * pow(1.0 - pow(ro, 2.0), 0.5));

	return val;
}

/********** Product of pdfs of a k iid normals N(mu[i],sigma) ******/

double pdf_delta(double *delta, double *mu, double sigma, int k)
{
	int i;
	double a;

	a = 1.0;
	for (i = 0; i < k; i++)
		a = a * pdf_norm(delta[i], mu[i], sigma);

	return a;
}

double PhiErf(double x)
{
	double y;

	y = 0.5 * (1.0 + erf(x / sqrt2));

	return y;
}

double FqCauchy(double x)
{
	double val;

	val = (x / (2 * (1.0 + pow(x, 2.0))) + atan(x) / 2 + miPi / 4) / (miPi / 2);

	return val;
}

double sumsqFqCauchy(double *x, double *u, double *z1, double **z2, double **z3)
{
	double val;

	val = pow(FqCauchy(x[0]) - u[0], 2.0);

	return val;
}

#define A 12
double sp_gamma(double z)
{
	const int a = A;
	static double c_space[A];
	static double *c = NULL;
	int k;
	double accm;

	if (c == NULL)
	{
		double k1_factrl = 1.0; /* (k - 1)!*(-1)^k with 0!==1*/
		c = c_space;
		c[0] = sqrt(2.0 * miPi);
		for (k = 1; k < a; k++)
		{
			c[k] = exp(a - k) * pow(a - k, k - 0.5) / k1_factrl;
			k1_factrl *= -k;
		}
	}
	accm = c[0];
	for (k = 1; k < a; k++)
	{
		accm += c[k] / (z + k);
	}
	accm *= exp(-(z + a)) * pow(z + a, z + 0.5); /* Gamma(z+1) */
	return accm / z;
}

void simyt(double *x, double *rt)
{
	int i;
	double ht, ht1, mu, phi, alpha, gamma1, gamma2, sigma, sigma2, phi2, eps, feps;

	mu = x[0];
	phi = x[1];
	alpha = x[2];
	gamma1 = x[3];
	gamma2 = x[4];
	sigma2 = fabs(x[5]);

	phi2 = pow(phi, 2.0);
	sigma = sqrt(sigma2);

	ht1 = mu + sigma * rnor() / sqrt(1.0 - phi2);

	for (i = 0; i < 10; i++)
	{
		ht = ht1;
		eps = rnor();
		feps = alpha * (eps < 0.0 ? 1.0 : 0.0) + gamma1 * eps + gamma2 * (fabs(eps) - sqrt2dPi);
		ht1 = mu + phi * (ht - mu) + feps + sigma * rnor();
	}

	for (i = 0; i < l; i++)
	{
		ht = ht1;
		eps = rnor();
		feps = alpha * (eps < 0.0 ? 1.0 : 0.0) + gamma1 * eps + gamma2 * (fabs(eps) - sqrt2dPi);
		ht1 = mu + phi * (ht - mu) + feps + sigma * rnor();

		rt[i] = exp(ht / 2.0) * eps;
	}
}

int compare(const void *a, const void *b)
{
	if (hsV[*(int *)a] > hsV[*(int *)b])
		return 1;
	else if (hsV[*(int *)a] < hsV[*(int *)b])
		return -1;
	else
		return 0;
}

double rhoc(double z, double tcc)
{
	if (z < (-tcc))
		return (tcc * log((z + tcc) / tcc + sqrt(1.0 + pow((z + tcc) / tcc, 2.0))) - tcc);
	else
		return z;
}

double Frob(double x)
{
	double a, Ac, Ktxt;

	if (x < (-csqtr))
		a = D1p / (ctrunc - 1.0) * pow(fabs(x), 1.0 - ctrunc);
	else
	{
		Ac = csqtr * pd1 / (ctrunc - 1.0);
		if (x < (csqtr))
			a = Ac + PhiErf(x) - PhiErf(-csqtr);
		else
			a = 2 * Ac + 2 * PhiErf(csqtr) - 1.0 + D1p * pow(x, 1.0 - ctrunc) / (1.0 - ctrunc);
	}
	Ktxt = 1.0 / (pdf_norm(-csqtr, 0.0, 1.0) * fabs(-csqtr) / (ctrunc - 1.0) + phiconst + pdf_norm(csqtr, 0.0, 1.0) * fabs(csqtr) / (ctrunc - 1.0));
	a = a * Ktxt;

	return a;
}

double PF(double *x, double *rt, double *pt, double **z2, double **z3)
{
	int i0, i, j, k, t, cont, Hr, kcont, imin, imax, contres, contres0;
	double ma50, ma250, logfrt, skewp, yp, ym, Ktxt, delta, D1, D2, sigmaxt, driftmu, beta1, beta2, beta3, beta4, beta5, mud, sigmaxi, gam1nu, val0, sqrtnu, EabsE, C0, nu, s, bandc, ht, mu, phi, alpha, gamma1, gamma2, sigma, sigma2, phi2, eps, feps, cond, a, val, yb, Like, lambda0, num, den, fnp, varind, sigmadi, Meanrt, Cpi, Ur, Wr, h2, up;
	double *Probi, *piMalik;
	double **hs, **hss;
	int *sigmacount;

	jz, jsr = 123456789;

	seed = 1111;

	zigset(seed);
	srand(seed);

	hs = alocmat(B, 2);
	Probi = alocvec(B);
	piMalik = alocvec(B + 1);
	sigmacount = alocintvec(B);

	mu = x[0];
	phi = x[1];
	sigma = x[2];
	mud = x[3];

	phi2 = pow(phi, 2.0);
	sigma2 = sigma * sigma;

	for (cont = 0; cont < B; cont++)
		hs[cont][1] = mu / (1.0 - phi) + sqrt(sigma2 / (1.0 - phi2)) * rnor();

	Like = 0.0;

	i0 = 0;

	for (t = i0; t < l; t++)
	{

		val = 0.0;

		for (i = 0; i < B; i++)
			sigmacount[i] = i;

		for (cont = 0; cont < B; cont++)
		{

			/******** Step 1 *******************/

			hsV[cont] = hs[cont][1] = mu + phi * hs[cont][1] + sigma * rnor();

			sigmaxt = exp(hs[cont][1] / 2);

			driftmu = mud;

			Probi[cont] = pdf_norm(rt[t], driftmu, sigmaxt);
		}

		val = 0.0;
		for (cont = 0; cont < B; cont++)
			val += Probi[cont];

		qsort(sigmacount, B, sizeof(int), compare);

		for (cont = 0; cont < B; cont++)
		{
			hsV[cont] = hs[sigmacount[cont]][1];
			hs[cont][0] = Probi[sigmacount[cont]];
		}

		for (cont = 0; cont < B; cont++)
		{
			Probi[cont] = hs[cont][0];
			hs[cont][1] = hsV[cont];
		}

		if (fabs(val) > 0.0)
		{
			Like += log(val / B);

			for (cont = 0; cont < B; cont++)
				Probi[cont] = Probi[cont] / val;

			piMalik[0] = Probi[0] / 2;
			piMalik[B] = Probi[B - 1] / 2;
			for (cont = 1; cont < B; cont++)
				piMalik[cont] = (Probi[cont] + Probi[cont - 1]) / 2;

			/* Generating from the multinomial distribution, using Malmquist ordered statistics */

			contres0 = B;
			a = 1.0;
			for (cont = 0; cont < B; cont++)
			{
				a = pow(uni(), 1.0 / contres0) * a;
				hsV[B - cont - 1] = a;
				contres0 = contres0 - 1;
			}

			s = 0.0;
			j = 0;
			for (cont = 0; cont < (B + 1); cont++)
			{
				s += piMalik[cont];
				while ((hsV[j] <= s) && (j < B))
				{
					sigmacount[j] = cont;
					hs[j][0] = (hsV[j] - (s - piMalik[cont])) / piMalik[cont];
					j = j + 1;
				}
			}

			for (cont = 0; cont < B; cont++)
			{
				if (sigmacount[cont] == 0)
					hsV[cont] = hs[0][1];
				else
				{
					if (sigmacount[cont] == B)
						hsV[cont] = hs[B - 1][1];
					else
						hsV[cont] = (hs[sigmacount[cont]][1] - hs[sigmacount[cont] - 1][1]) * hs[cont][0] + hs[sigmacount[cont] - 1][1];
				}
			}

			for (cont = 0; cont < B; cont++)
				hs[cont][1] = hsV[cont];
		}
		else
		{
			Like = -BIG;
			t = l + 1;
		}
	}

	freemat(hs, B);
	free(Probi);
	free(sigmacount);
	free(piMalik);

	return -Like;
}

double quickselect(int k, int n, double *arr)
{
	int i, ir, j, ls, mid;
	double a, temp;

	ls = 0;
	ir = n - 1;
	for (;;)
	{
		if (ir <= ls + 1)
		{
			if (ir == ls + 1 && arr[ir] < arr[ls])
			{
				SWAP(arr[ls], arr[ir])
			}
			return arr[k];
		}
		else
		{
			mid = (ls + ir) >> 1;
			SWAP(arr[mid], arr[ls + 1])
			if (arr[ls + 1] > arr[ir])
			{
				SWAP(arr[ls + 1], arr[ir])
			}
			if (arr[ls] > arr[ir])
			{
				SWAP(arr[ls], arr[ir])
			}
			if (arr[ls + 1] > arr[ls])
			{
				SWAP(arr[ls + 1], arr[ls])
			}
			i = ls + 1;
			j = ir;
			a = arr[ls];
			for (;;)
			{
				do
					i++;
				while (arr[i] < a);
				do
					j--;
				while (arr[j] > a);
				if (j < i)
					break;
				SWAP(arr[i], arr[j])
			}
			arr[ls] = arr[j];
			arr[j] = a;
			if (j >= k)
				ir = j - 1;
			if (j <= k)
				ls = i;
		}
	}
}

int main(void)
{
	int i, j, j1, k, n, t, cont, jcont, trials, itno;
	double ftry, ftry1, ftry2, tol, a, sigmad0, val, eps1, sig, minval;
	double *pt, *mus, *muss, *tetaCons, *lambda;
	double **Pf, **test, **StdAux, **tetainit, **rts;

	n = 250;

	FILE *fp;
	char s[30], outputFile[100];

	l = 24050;
	lags = 250;

	B = 1000;
	trials = 20;
	itno = 1;

	sqrt2dPi = sqrt(2.0 / miPi);
	csqtr = sqrt(ctrunc);
	sqrt2 = sqrt(2.0);
	pd1 = exp(-ctrunc / 2) / sqrt(2 * miPi);
	D1p = pow(fabs(-csqtr), ctrunc) * pd1;
	phiconst = 1.0 - 2 * PhiErf(-csqtr);
	lbound = 1.0;
	logsqrt2pi = log(sqrt(2.0 * miPi));

	/*printf("phi=%.12f\n",phiconst);*/

	muaux = alocvec(pdim);
	tetaCons = alocvec(pdim);
	tetainit = alocmat((itno * trials), pdim);
	rts = alocmat(itno, l);
	pt = alocvec(l);
	hsV = alocvec(B);
	lambda = alocvec(pdim);
	mus = alocvec(B);
	muss = alocvec(B);
	monthmut = alocvec(l);
	shortmut = alocvec(l);
	driftmut = alocvec(l);
	ma50t = alocvec(l);
	ma250t = alocvec(l);

	muaux[0] = -0.190749834;
	muaux[1] = 0.977385697;
	muaux[2] = 0.215634224;
	muaux[3] = 0.015011035;

	seed = 1;

	zigset(seed);
	srand(seed);

	for (i = 0; i < (itno * trials); i++)
	{
		for (j = 0; j < pdim; j++)
			tetainit[i][j] = muaux[j] * (0.99 + 0.02 * uni());
	}

	fp = fopen("GSPC19280104-20230929.txt", "r");

	j = 0;

	while (fgets(s, 30, fp) != NULL)
	{
		if (j >= 0)
			rts[0][j] = atof(s);
		j++;
	}
	fclose(fp);

	sprintf(outputFile, "SV-T%dN%d_out.txt", l, B);

	fp = fopen(outputFile, "a");
	fprintf(fp, "%s	%s	%s	%s	%s\n", "a", "b", "sigma", "mu", "val");
	fclose(fp);

	lambda[0] = -0.5;
	lambda[1] = -0.02;
	lambda[2] = 0.2;
	lambda[3] = 0.01;

	minval = BIG;
	for (i = 0; i < pdim; i++)
		tetaCons[i] = tetainit[0][i];

	for (jcont = 0; jcont < trials; jcont++)
	{
		test = nlminLvectSimplex((*PF), tetainit[jcont], 2000, lambda, rts[0], pt, Weightmatrix, Weightmatrix, EPS1, pdim);
		test[0][2] = fabs(test[0][2]);

		if (test[0][pdim] < minval)
		{
			for (i = 0; i < pdim; i++)
				tetaCons[i] = test[0][i];
			minval = test[0][pdim];
		}

		fp = fopen(outputFile, "a");
		for (i = 0; i < pdim; i++)
			fprintf(fp, "%.12f	", test[0][i]);
		fprintf(fp, "%.16f\n", test[0][pdim]);
		fclose(fp);

		/*	printf("jcont=%d\n",jcont);
			writemat("t",pdim+1,pdim+1,test);*/

		freemat(test, pdim + 1);
	}

	free(muaux);
	freemat(rts, itno);
	free(tetaCons);
	freemat(tetainit, (itno * trials));
	free(hsV);
	free(lambda);
	free(mus);
	free(muss);
	free(pt);
	free(monthmut);
	free(shortmut);
	free(driftmut);
	free(ma50t);
	free(ma250t);

	exit(0);
}
