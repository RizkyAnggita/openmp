// serial.c
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NMAX 100
#define DATAMAX 1000
#define DATAMIN -1000

/* 
 * Struct Matrix
 *
 * Matrix representation consists of matrix data 
 * and effective dimensions 
 * */
typedef struct Matrix {
	int mat[NMAX][NMAX];	// Matrix cells
	int row_eff;			// Matrix effective row
	int col_eff;			// Matrix effective column
} Matrix;


/* 
 * Procedure init_matrix
 * 
 * Initializing newly allocated matrix
 * Setting all data to 0 and effective dimensions according
 * to nrow and ncol 
 * */
void init_matrix(Matrix *m, int nrow, int ncol) {
	m->row_eff = nrow;
	m->col_eff = ncol;

	for (int i = 0; i < m->row_eff; i++) {
		for (int j = 0; j < m->col_eff; j++) {
			m->mat[i][j] = 0;
		}
	}
}


/* 
 * Function input_matrix
 *
 * Returns a matrix with values from stdin input
 * */
Matrix input_matrix(int nrow, int ncol) {
	Matrix input;
	init_matrix(&input, nrow, ncol);

	for (int i = 0; i < nrow; i++) {
		for (int j = 0; j < ncol; j++) {
			scanf("%d", &input.mat[i][j]);
		}
	}

	return input;
}


/* 
 * Procedure print_matrix
 * 
 * Print matrix data
 * */
void print_matrix(Matrix *m) {
	for (int i = 0; i < m->row_eff; i++) {
		for (int j = 0; j < m->col_eff; j++) {
			printf("%d ", m->mat[i][j]);
		}
		printf("\n");
	}
}


/* 
 * Function get_matrix_datarange
 *
 * Returns the range between maximum and minimum
 * element of a matrix
 * */
int get_matrix_datarange(Matrix *m) {
	int max = DATAMIN;
	int min = DATAMAX;
	for (int i = 0; i < m->row_eff; i++) {
		for (int j = 0; j < m->col_eff; j++) {
			int el = m->mat[i][j];
			if (el > max) max = el;
			if (el < min) min = el;
		}
	}

	return max - min;
}


/*
 * Function supression_op
 *
 * Returns the sum of intermediate value of special multiplication
 * operation where kernel[0][0] corresponds to target[row][col]
 * */
int supression_op(Matrix *kernel, Matrix *target, int row, int col) {
	int sum = 0;
	int i;

    #pragma omp parallel for reduction(+:sum)
        for (i = 0; i < kernel->row_eff; i++) {


            for (int j = 0; j < kernel->col_eff; j++) {
				int nthreads, tid;
				nthreads = omp_get_num_threads();
				tid = omp_get_thread_num();
				printf("Hello world from  threadId %d out of %d threads\n", tid, nthreads);
                sum += kernel->mat[i][j] * target->mat[row + i][col + j];
				printf("Hasil sum: %d target[%d + %d][%d + %d]\n", sum, row, i, col, j);

            }
                // #pragma omp critical
                // {
                //     intermediate_sum += sum_i;
                // }
        }

	return sum;
}


/* 
 * Function convolution
 *
 * Return the output matrix of convolution operation
 * between kernel and target
 * */
Matrix convolution(Matrix *kernel, Matrix *target) {
	Matrix out;
	int out_row_eff = target->row_eff - kernel->row_eff + 1;
	int out_col_eff = target->col_eff - kernel->col_eff + 1;
	
	init_matrix(&out, out_row_eff, out_col_eff);

	#pragma omp parallel for num_threads(8) collapse(2)
	for (int i = 0; i < out.row_eff; i++) {
		for (int j = 0; j < out.col_eff; j++) {
			out.mat[i][j] = supression_op(kernel, target, i, j);
			printf("Out.mat[%d][%d]: %d\n", i, j, out.mat[i][j]);
		}
	}

	return out;
}


/*
 * Procedure merge_array
 *
 * Merges two subarrays of n with n[left..mid] and n[mid+1..right]
 * to n itself, with n now ordered ascendingly
 * */
void merge_array(int *n, int left, int mid, int right) {
	int n_left = mid - left + 1;
	int n_right = right - mid;
	int iter_left = 0, iter_right = 0, iter_merged = left;
	int arr_left[n_left], arr_right[n_right];

	for (int i = 0; i < n_left; i++) {
		arr_left[i] = n[i + left];
	}

	for (int i = 0; i < n_right; i++) {
		arr_right[i] = n[i + mid + 1];
	}

	while (iter_left < n_left && iter_right < n_right) {
		if (arr_left[iter_left] <= arr_right[iter_right]) {
			n[iter_merged] = arr_left[iter_left++];
		} else {
			n[iter_merged] = arr_right[iter_right++];
		}
		iter_merged++;
	}

	while (iter_left < n_left)  {
		n[iter_merged++] = arr_left[iter_left++];
	}
	while (iter_right < n_right) {
		n[iter_merged++] = arr_right[iter_right++];
	} 
}


/* 
 * Procedure merge_sort
 *
 * Sorts array n with merge sort algorithm
 * */
void merge_sort(int *n, int left, int right) {
	if (left < right) {
		int mid = left + (right - left) / 2;

		merge_sort(n, left, mid);
		merge_sort(n, mid + 1, right);

		merge_array(n, left, mid, right);
	}	
}
 

/* 
 * Procedure print_array
 *
 * Prints all elements of array n of size to stdout
 * */
void print_array(int *n, int size) {
	for (int i = 0; i < size; i++ ) printf("%d ", n[i]);
	printf("\n");
}


/* 
 * Function get_median
 *
 * Returns median of array n of length
 * */
int get_median(int *n, int length) {
	int mid = length / 2;
	if (length & 1) return n[mid];

	return (n[mid - 1] + n[mid]) / 2;
}


/* 
 * Function get_floored_mean
 *
 * Returns floored mean from an array of integers
 * */
long get_floored_mean(int *n, int length) {
	long sum = 0;
	for (int i = 0; i < length; i++) {
		sum += n[i];
	}

	return sum / length;
}



// main() driver
int main() {
	int kernel_row, kernel_col, target_row, target_col, num_targets;
	struct timespec start, stop;

	// reads kernel's row and column and initalize kernel matrix from input
	scanf("%d %d", &kernel_row, &kernel_col);
	Matrix kernel = input_matrix(kernel_row, kernel_col);
	
	// reads number of target matrices and their dimensions.
	// initialize array of matrices and array of data ranges (int)
	scanf("%d %d %d", &num_targets, &target_row, &target_col);
	Matrix* arr_mat = (Matrix*)malloc(num_targets * sizeof(Matrix));
	int arr_range[num_targets];
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	// read each target matrix, compute their convolution matrices, and compute their data ranges
	for (int i = 0; i < num_targets; i++) {
		arr_mat[i] = input_matrix(target_row, target_col);
		// arr_mat[i] = convolution(&kernel, &arr_mat[i]);
		Matrix out;
		int out_row_eff = arr_mat[i].row_eff - kernel.row_eff + 1;
		int out_col_eff = arr_mat[i].col_eff - kernel.col_eff + 1;
		
		init_matrix(&out, out_row_eff, out_col_eff);

		#pragma omp parallel for num_threads(8) collapse(2)
		for (int j = 0; j < out.row_eff; j++) {
			for (int k = 0; k < out.col_eff; k++) {
				// out.mat[i][j] = supression_op(kernel, target, i, j);
				int sumSupress = 0;
				int l;
				int rowSupress = j;
				int colSupress = k;

				#pragma omp parallel for reduction(+:sumSupress)
				for (l = 0; l < kernel.row_eff; l++) {

					for (int m = 0; m< kernel.col_eff; m++) {
						int nthreads, tid;
						nthreads = omp_get_num_threads();
						tid = omp_get_thread_num();
						printf("Hello world from  threadId %d out of %d threads\n", tid, nthreads);
						sumSupress += kernel.mat[l][m] * arr_mat[l].mat[rowSupress + l][colSupress + m];
						printf("Hasil sum: %d target[%d + %d][%d + %d]\n", sumSupress, rowSupress, l, colSupress, m);

					}
						// #pragma omp critical
						// {
						//     intermediate_sum += sum_i;
						// }
				}
				printf("Out.mat[%d][%d]: %d\n",j, k, out.mat[j][k]);
				out.mat[j][k] = sumSupress;
			}
		}
		arr_mat[i] = out;
		arr_range[i] = get_matrix_datarange(&arr_mat[i]); 
	}

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	double result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
	printf("Time Elapsed: %.8f\n", result);
	// sort the data range array
	merge_sort(arr_range, 0, num_targets - 1);

	for (int i = 0; i < num_targets; i++)
	{
		printf("MERGE SORT[%d]: %d\n", i, arr_range[i]);
	}
	
	
	int median = get_median(arr_range, num_targets);	
	int floored_mean = get_floored_mean(arr_range, num_targets); 

	// print the min, max, median, and floored mean of data range array
	printf("%d\n%d\n%d\n%d\n", 
			arr_range[0], 
			arr_range[num_targets - 1], 
			median, 
			floored_mean);

	
	return 0;
}