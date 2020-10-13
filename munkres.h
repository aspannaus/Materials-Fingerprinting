/* File: munkres.h
 * Date: 08/03/2018
 * Author: Adam Spannaus
 * Description: header for hungarian algorithm
 */
#include <stdbool.h>
#ifndef MUNKRES_H
#define MUNKRES_H


#define TOL 0.0001


typedef struct idx_t { int row;
                       int col;
                       int ROWS;
                       int COLS;
                       int k;
                       int dims;
                       int nrows; // rows - m
                       int ncols; // cols - n
                       int path_ct;
                       double min;
                       double **dm;
                       int **M;
                       int** path;
                       int* path_row;
                       int* path_col;
                       int* row_cover;
                       int* col_cover;
                       int path_count;
                       int path_row_0;
                       int path_col_0;} idx_t;

double get_random ();
void print_mat( double**, int, int );
void print_matches( int**, int, int );
struct idx_t* initIdxs( double**, int, int );
int step_one( struct idx_t* );
int step_two( struct idx_t* );
int step_three( struct idx_t* );
int step_four( struct idx_t* );
int step_five( struct idx_t* );
int step_six( struct idx_t* );
void find_zero( struct idx_t* );
bool star_in_row( struct idx_t* );
void find_star_in_row( struct idx_t* );
void find_star_in_col( struct idx_t* );
void find_prime_in_row( struct idx_t* );
void augment_path( struct idx_t* );
void clear_covers( struct idx_t* );
void erase_primes( struct idx_t* );
void min_val( struct idx_t* );
double calc_cost( double**, struct idx_t* );
double assignment( double**, int, int );
void deleteMem( struct idx_t* );

#endif
