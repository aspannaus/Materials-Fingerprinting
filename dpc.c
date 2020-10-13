#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "munkres.h"


double inf_norm (const double* restrict x, const double* restrict y)
/*
    returns l^infty norm between x - y, where x and y are in R^2

 */
{
        double tmp1 = fabs(x[0] - y[0]);
        double tmp2 = fabs(x[1] - y[1]);
        return tmp1 > tmp2 ? tmp1 : tmp2; // max
}

double dist_fun (const double* restrict x, const double* restrict y, double eps, double p)
/*
    returns first part of dpc distance
 */
{
        double tmp;
        double tmp1;
        double d;

        d = inf_norm(x, y);
        tmp = d > eps ? eps : d; // min
        if (p == 1) {
                tmp1 = tmp;
        } else if (p == 2) {
                tmp1 = tmp * tmp;
        } else {
                tmp1 = pow (tmp, p);
        }
        return tmp1;
}


double dpc (const double** restrict d1, const double** restrict d2, double p, double eps,
            int ROWS, int COLS){
        double cost = 0;
        int i;
        int j;

        double **dm = (double**)( malloc( ROWS * sizeof( double* )));
        dm[0] = (double*)(malloc( ROWS * COLS * sizeof( double )));

        for ( i = 1; i < ROWS; i++ )
                dm[i] = dm[i-1] + COLS;

        // #pragma omp parallel for collapse(2) private(j) schedule(static)
        for ( i = 0; i < ROWS; i++ ) {
                for ( j = 0; j < COLS; j++ ) {
                        dm[i][j] = dist_fun(d1[i], d2[j], eps, p);
                }
        }

        cost = assignment(dm, ROWS, COLS); // from Munkres
        cost += (COLS - ROWS) * pow(eps, p);
        cost = (1.0 / COLS) * cost;
        if (p == 2) {
                cost = sqrt (cost);
        } else if (p > 2) {
                cost = pow (cost, 1.0 / p );
        }
        free(dm[0]);
        free(dm);

        return cost;
}
