#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <math.h>
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

#define A 12

int compare(const void *a, const void *b)
{
	if (hsV[*(int *)a] > hsV[*(int *)b])
		return 1;
	else if (hsV[*(int *)a] < hsV[*(int *)b])
		return -1;
	else
		return 0;
}

double PF(double *x, double *rt, double *pt, double **z2, double **z3)
{
	int i0, i, j, k, t, cont, Hr, kcont, imin, imax, contres, contres0;
	double ma50, ma250, logfrt, skewp, yp, ym, Ktxt, delta, D1, D2, sigmaxt, driftmu, beta1, beta2, beta3, beta4, beta5, mud, sigmaxi, gam1nu, val0, sqrtnu, EabsE, C0, nu, s, bandc, ht, mu, phi, alpha, gamma1, gamma2, sigma, sigma2, phi2, eps, feps, cond, a, val, yb, Like, lambda0, num, den, fnp, varind, sigmadi, Meanrt, Cpi, Ur, Wr, h2, up;
	double *Probi, *piMalik;
	double **hs, **hss;
	int *sigmacount;

	jz, jsr = 123456789;
	// printf("Value of jz is: %i\n", jz);
	// printf("Value of jsr is: %i\n", jsr);

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

int main(void)
{
	int i, j, j1, k, n, t, cont, jcont, trials, itno;
	double ftry, ftry1, ftry2, tol, a, sigmad0, val, eps1, sig, minval;
	double *pt, *mus, *muss, *tetaCons, *lambda;
	double **Pf, **test, **StdAux, **tetainit, **rts;

	n = 250;

	FILE *fp;
	char s[30], outputFile[100];

	l = 24050; // Aantal regels van GSPC
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

	for (i = 0; i < (itno * trials); i++) // 20
	{
		for (j = 0; j < pdim; j++) // 4, dus 80 keer in totaal
			tetainit[i][j] = muaux[j] * (0.99 + 0.02 * uni()); // Dit gebruikt weer de ziggurat method
	}

	fp = fopen("GSPC19280104-20230929.txt", "r");

	j = 0;

	while (fgets(s, 30, fp) != NULL) // Maakt een matrix van GSPC
	{
		if (j >= 0)
			rts[0][j] = atof(s); // rts is een matrix van 24050 x 1
		j++;
	}
	fclose(fp);

	sprintf(outputFile, "SV-T%dN%d_out.txt", l, B); // Generates the output files with enough lines

	fp = fopen(outputFile, "a");
	fprintf(fp, "%s	%s	%s	%s	%s\n", "a", "b", "sigma", "mu", "val"); // Sets "a b sigma mu val" in top row of the file
	fclose(fp);

	lambda[0] = -0.5;
	lambda[1] = -0.02;
	lambda[2] = 0.2;
	lambda[3] = 0.01;

	minval = BIG; // Big heeft een waarde van 1.0e30
	for (i = 0; i < pdim; i++)
		tetaCons[i] = tetainit[0][i]; // tetaCons is een vector van 4, tetainit is een matrix van 4x20

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