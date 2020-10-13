#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <stdbool.h>
#include <time.h>
#include "munkres.h"


double cpu_time (void)
/*
    returns the current reading on the clock.
    Output, double the current reading of the clock in seconds.
 */
{
        double value;

        value = ( double ) clock ( )
                / ( double ) CLOCKS_PER_SEC;

        return value;
}

void print_matches( int** M, int ROWS, int COLS )
{
        int i;
        int j;

        printf( "[" );
        for ( i = 0; i < ROWS; i++ ) {
                printf( " %2d ", i );
        }
        printf( "]\n[" );

        for ( i = 0; i < ROWS; i++ ) {
                for ( j = 0; j < COLS; j++ ) {
                        if ( M[i][j] == 1 ) {
                                printf( " %2d ", j );
                        }
                }
        }
        printf( "]\n" );
}

void print_mat( double** A, int rows, int cols )
{
        size_t i, j;
        printf("[");
        for (i = 0; i < rows; i++) {
                printf("[");
                for (j = 0; j < cols; j++ ) {
                        printf("%10.8lf  ",  A[i][j]);
                }
                printf("]\n");
        }
}


double get_random() {
        return ((double)rand() / (double)RAND_MAX);
}

struct idx_t* initIdxs( double** restrict C, int ROWS, int COLS )
{
        int i;
        int j;
        struct idx_t* idxs = malloc( sizeof( struct idx_t ));
        assert( idxs != NULL );

        int dims = ROWS > COLS ? ROWS : COLS;
        idxs->k = ROWS < COLS ? ROWS : COLS;
        idxs->ROWS = ROWS;
        idxs->COLS = COLS;

        idxs->ncols = dims; // columns - n

        idxs->min = DBL_MAX;
        idxs->dm = calloc(dims, sizeof( double* ));
        idxs->dm[0] = calloc(dims * dims, sizeof( double ));

        idxs->M = calloc(dims, sizeof( int* ));
        idxs->M[0] = calloc( dims * dims, sizeof( int ));

        idxs->path = calloc(( dims + dims ), sizeof( int* ));
        idxs->path[0] = calloc(2 * ( dims + dims ), sizeof( int* ));

        idxs->row_cover = calloc( dims, sizeof( int ));
        idxs->col_cover = calloc( dims, sizeof( int ));
        idxs->path_row = calloc( dims, sizeof( int ));
        idxs->path_col = calloc( dims, sizeof( int ));

        for (i = 1; i < dims; i++) {
                idxs->dm[i] = idxs->dm[i-1] + dims;
                idxs->M[i] = idxs->M[i-1] + dims;
        }

        for (i = 1; i < dims + dims; i++ ) {
                idxs->path[i] = idxs->path[i-1] + 2;
        }
        for ( i = 0; i < ROWS; i++ ) {
                for ( j = 0; j < COLS; j++ ) {
                        idxs->dm[i][j] = C[i][j];
                }
        }

        return idxs;
}

// For each row of the cost matrix, find the smallest element and subtract
// it from every element in its row.
int step_one( struct idx_t* idxs ) {
        int r, c;
        double min_in_row;

        for (r = 0; r < idxs->nrows; r++ ) {
                min_in_row = idxs->dm[r][0];
                for (c = 0; c < idxs->ncols; c++) {
                        if (idxs->dm[r][c] < min_in_row) {
                                min_in_row = idxs->dm[r][c];
                        }
                }
                for (c = 0; c < idxs->ncols; c++) {
                        idxs->dm[r][c] -= min_in_row;
                }
        }
        return 2;
}

//Find a zero (Z) in the resulting matrix.  If there is no starred
//zero in its row or column, star Z. Repeat for each element, then go to Step 3.
int step_two( struct idx_t* idxs ){
        int r, c;

        for (r = 0; r < idxs->nrows; r++) {
                for (c = 0; c < idxs->ncols; c++) {
                        if (idxs->dm[r][c] < TOL) {
                                if (idxs->row_cover[r] == 0 && idxs->col_cover[c] == 0 ) {
                                        idxs->M[r][c] = 1;
                                        idxs->dm[r][c] = 0.0;
                                        idxs->row_cover[r] = 1;
                                        idxs->col_cover[c] = 1;
                                }
                        }
                }
        }

        for (r = 0; r < idxs->nrows; r++) {
                idxs->row_cover[r] = 0;
        }
        for (c = 0; c < idxs->ncols; c++) {
                idxs->col_cover[c] = 0;
        }
        return 3;
}

//Cover each column containing a starred zero.  If all columns are covered,
//these  describe a complete set of unique assignments, and we're done.
// Otherwise, go to Step 4.
int step_three( struct idx_t* idxs ){
        int r, c;
        int col_count = 0;
        int step = 0;

        for (r = 0; r < idxs->nrows; r++ ) {
                for (c = 0; c < idxs->ncols; c++ ) {
                        if ( idxs->M[r][c] == 1 ) {
                                idxs->col_cover[c] = 1;
                                col_count++;
                        }
                }
        }

        if (col_count == idxs->ncols) {
                step = 7;
        } else {
                step = 4;
        }
        return step;
}

void find_zero( struct idx_t* idxs ){
        int r = 0;
        int c = 0;
        bool done = false;
        idxs->row = -1;
        idxs->col = -1;

        while (!done) {
                c = 0;
                while (true) {
                        if (idxs->dm[r][c] < TOL) {
                                if (idxs->row_cover[r] == 0 && idxs->col_cover[c] == 0) {
                                        idxs->row = r;
                                        idxs->col = c;
                                        idxs->dm[r][c] = 0;
                                        done = true;
                                }
                        }
                        c++;
                        if (c >= idxs->ncols || done) {
                                break;
                        }
                }
                r++;
                if (r >= idxs->nrows) {
                        done = true;
                }
        }
}
bool star_in_row( struct idx_t* idxs ){
        bool tmp = false;
        int c = 0;

        for (c = 0; c < idxs->ncols; c++) {
                if (idxs->M[idxs->row][c] == 1) {
                        tmp = true;
                        idxs->col = c;
                        break;
                }
        }
        return tmp;
}
void find_star_in_row( struct idx_t* idxs ){
        int c = 0;
        idxs->col = -1;

        for ( c = 0; c < idxs->ncols; c++ ) {
                if ( idxs->M[idxs->row][c] == 1 ) {
                        idxs->col = c;
                        break;
                }
        }
}
//Find a noncovered zero and prime it.  If there is no starred zero
//in the row containing it, go to step 5.  Otherwise,
//cover this row and uncover the column containing the starred zero.
//Continue until there are no uncovered zeros.
//Save the min uncovered value and go to Step 6.
int step_four( struct idx_t* idxs ){
        int step = 1;
        bool done = false;
        idxs->row = -1;
        idxs->col = -1;

        while (!done) {
                find_zero( idxs );
                if ( idxs->row == -1 ) {
                        done = true;
                        step = 6;
                } else {
                        idxs->M[idxs->row][idxs->col] = 2;
                        if (star_in_row( idxs )) {
                                idxs->row_cover[idxs->row] = 1;
                                idxs->col_cover[idxs->col] = 0;
                        } else {
                                done = true;
                                step = 5;
                                idxs->path_row_0 = idxs->row;
                                idxs->path_col_0 = idxs->col;
                        }
                }
        }
        return step;
}
void find_star_in_col( struct idx_t* idxs ){
        int i;
        int col = idxs->path[idxs->path_ct][1];
        idxs->row = -1;

        for (i = 0; i < idxs->nrows; i++) {
                if ( idxs->M[i][col] == 1 )
                        idxs->row = i;
        }
}
void find_prime_in_row( struct idx_t* idxs ){
        int i;
        int row = idxs->path[idxs->path_ct][0];

        for (i = 0; i < idxs->ncols; i++) {
                if (idxs->M[row][i] == 2)
                        idxs->col = i;
        }
        if (idxs->M[row][idxs->col] != 2) {
                idxs->col = -1;
        }
}
void augment_path( struct idx_t* idxs ){
        int p;

        for (p = 0; p < idxs->path_ct+1; p++) {
                if ( idxs->M[idxs->path[p][0]][idxs->path[p][1]] == 1 ) {
                        idxs->M[idxs->path[p][0]][idxs->path[p][1]] = 0;
                } else {
                        idxs->M[idxs->path[p][0]][idxs->path[p][1]] = 1;
                }
        }
}
void clear_covers( struct idx_t* idxs ){
        int r;
        int c;

        for (r = 0; r < idxs->nrows; r++) {
                idxs->row_cover[r] = 0;
        }
        for (c = 0; c < idxs->ncols; c++) {
                idxs->col_cover[c] = 0;
        }

}
void erase_primes( struct idx_t* idxs ){
        int r, c;

        for (r = 0; r < idxs->nrows; r++) {
                for (c = 0; c < idxs->ncols; c++) {
                        if (idxs->M[r][c] == 2) {
                                idxs->M[r][c] = 0;
                        }
                }
        }
}
//Construct a series of alternating primed and starred zeros.
//Z0 is the uncovered primed zero found in Step 4.  Z1 is
//the starred zero in the column of Z0 (if any). Z2 is the primed zero
//in the row of Z1. Continue until the series
//ends at a primed zero that has no starred zero in its column.
//Clear each starred zero of the series, star each primed zero of the series.
//Then erase all primes and uncover every line in the matrix. Return to Step 3.
int step_five( struct idx_t* idxs ) {
        bool done = false;
        int tmp;

        idxs->path_ct = 0;
        idxs->row = -1;
        idxs->col = -1;
        idxs->path[0][0] = idxs->path_row_0;
        idxs->path[0][1] = idxs->path_col_0;

        while (!done) {
                find_star_in_col( idxs );
                if ( idxs->row > -1 ) {
                        idxs->path_ct++;
                        tmp = idxs->path_ct - 1;
                        idxs->path[idxs->path_ct][0] = idxs->row;
                        idxs->path[idxs->path_ct][1] = idxs->path[tmp][1];
                } else {
                        done = true;
                }
                if (!done) {
                        find_prime_in_row( idxs );
                        idxs->path_ct++;
                        tmp = idxs->path_ct - 1;
                        idxs->path[idxs->path_ct][0] = idxs->path[tmp][0];
                        idxs->path[idxs->path_ct][1] = idxs->col;
                }
        }
        augment_path( idxs );
        clear_covers( idxs );
        erase_primes( idxs );
        return 3;
}

void min_val( struct idx_t* idxs ) {
        int r, c;

        for (r = 0; r < idxs->nrows; r++) {
                for (c = 0; c < idxs->ncols; c++) {
                        if (idxs->col_cover[c] == 0  && idxs->row_cover[r] == 0) {
                                if (idxs->min > idxs->dm[r][c]) {
                                        idxs->min = idxs->dm[r][c];
                                }
                        }
                }
        }
}
//Add the value found in Step 4 to every element of each covered row, and subtract
//it from every element of each uncovered column.  Return to Step 4
int step_six( struct idx_t* idxs ) {
        int r;
        int c;

        min_val( idxs );

        for ( r = 0; r < idxs->nrows; r++ ) {
                for ( c = 0; c < idxs->ncols; c++ ) {
                        if ( idxs->row_cover[r] == 1 ) {
                                idxs->dm[r][c] += idxs->min;
                        }
                        if( idxs->col_cover[c] == 0 ) {
                                idxs->dm[r][c] -= idxs->min;
                        }
                }
        }
        return 4;
}


double calc_cost( double **C, struct idx_t* idxs ){
        int i;
        int j;
        double cost = 0;

        for (i = 0; i < idxs->ROWS; i++) {
                for (j = 0; j < idxs->COLS; j++) {
                        if ( idxs->M[i][j] == 1 ) {
                                cost += C[i][j];
                        }
                }
        }
        return cost;
}
void deleteMem(struct idx_t* idxs ){

        free( idxs->path_row );
        free( idxs->path_col );
        free( idxs->row_cover );
        free( idxs->col_cover );
        free( idxs->path[0]);
        free( idxs->path);
        free( idxs->dm[0] );
        free( idxs->dm );
        free( idxs->M[0] );
        free( idxs->M );
        free( idxs );
        return;
}
double assignment( double **C, int ROWS, int COLS ){
        bool done = false;
        int step = 1;
        double cost = 0;

        struct idx_t* idxs;
        idxs = initIdxs( C, ROWS, COLS );

        while (!done) {
                switch (step) {
                case 1:
                        step = step_one( idxs );
                        break;
                case 2:
                        step = step_two( idxs );
                        break;
                case 3:
                        step = step_three( idxs );
                        break;
                case 4:
                        step = step_four( idxs );
                        break;
                case 5:
                        step = step_five( idxs );
                        break;
                case 6:
                        step = step_six( idxs );
                        break;
                case 7:
                        done = true;
                        cost = calc_cost( C, idxs );
                        break;
                }
        }

        // print_matches( idxs->M, ROWS, COLS );
        deleteMem( idxs );
        return cost;
}

int main(){
        int i, j;
        unsigned int start;
        double diff;
        int m = 8; // rows
        int n = 15; // cols
        double c = 0;
        int dims = m > n ? m : n;
        double **C = calloc( dims, sizeof( double* ));
        C[0] = calloc( dims * dims, sizeof(double));
        int **M = calloc( dims, sizeof( int* ));
        M[0] = calloc( dims * dims, sizeof(int));

        for( i = 1; i < dims; i++ ) {
                C[i] = C[i-1] + dims;
                M[i] = M[i-1] + dims;
        }

        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) {
                        C[i][j] = (i + 1) * (j + 1)* 2 * get_random();
                }
        }

        start = cpu_time ( );

        c = assignment(C, dims, dims);

        // print_matches( M, m, n );

        printf("\nIn c, cost = %10.4lf\n", c);
        diff = cpu_time() - start;
        // msec = diff * (1000.0 / cps);
        // sec = diff / cps;
        printf("Total time: %e seconds\n", diff);

        free( C[0] );
        free( C );
        free( M[0] );
        free( M );


        return 0;
}
