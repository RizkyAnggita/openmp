// serial.c
#include <omp.h>
#include <mpi.h>
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
	Matrix* arr_mat;
	Matrix kernel;
	int kernel_row, kernel_col, target_row, target_col, num_targets;
	struct timespec start, stop;


	MPI_Init(NULL, NULL);
	printf("HEREE juga3\n");
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	printf("HEREE juga4\n");
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	int num_procs, max_threads;
	num_procs = omp_get_num_procs();
	printf("Num Procs: %d\n", num_procs);
	omp_set_num_threads(8);
	max_threads = omp_get_max_threads();
	printf("Max Thread: %d\n", max_threads);

	int matrix_target_per_process;
	int n_matrix_received;
	int sizeKernel;
	int arrRangeKonvo[num_targets];
	MPI_Status status;

	if (world_rank == 0) {
			printf("OTW MINTA INPUT\n");
	// reads kernel's row and column and initalize kernel matrix from input
		scanf("%d %d", &kernel_row, &kernel_col);
		kernel = input_matrix(kernel_row, kernel_col);
		sizeKernel = kernel_row * kernel_col;
		// reads number of target matrices and their dimensions.
		// initialize array of matrices and array of data ranges (int)
		scanf("%d %d %d", &num_targets, &target_row, &target_col);
		arr_mat = (Matrix*)malloc(num_targets * sizeof(Matrix));
		
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		// read each target matrix, compute their convolution matrices, and compute their data ranges
		printf("HEREE\n");
		for (int i = 0; i < num_targets; i++) {
			printf("UHUY: %d\n", i);
			arr_mat[i] = input_matrix(target_row, target_col);
			print_matrix(&arr_mat[i]);
		}
		printf("HEREE juga\n");
		int index, i;
		matrix_target_per_process = num_targets / world_size;
		printf("MATRIX TARGET PER PROCESS: %d\n", matrix_target_per_process);

		if (world_size > 1)	 {
			printf("WORLD SIZE > 1: world_size: %d\n", world_size);
			for (i = 1; i < world_size-1; i++)
			{
				index = i * matrix_target_per_process;
				printf("MSUK SINI i:%d index:%d\n", i, index);
				
				// MPI_Send(&sizeKernel, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&kernel, 10002,MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&matrix_target_per_process, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&arr_mat[index], matrix_target_per_process * 10002, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
			
			// MPI_Send(&sizeKernel, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&kernel, 10002, MPI_INT, i, 0, MPI_COMM_WORLD);

			// last process adds remaining elements
			index = i * matrix_target_per_process;
			printf("MSUK SINI i:%d index:%d\n", i, index);
			int matrix_left = num_targets - index;

			MPI_Send(&matrix_left, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&arr_mat[index], matrix_left * 10002, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
		// master process its own matrix
		printf("HERE BUY!!");
		int arr_range[matrix_target_per_process];
		for (int i = 0; i < matrix_target_per_process; i++) {
			// arr_mat[i] = input_matrix(target_row, target_col);
			// arr_mat[i] = convolution(&kernel, &arr_mat[i]);

			// Start of Convolution
			Matrix out;
			int out_row_eff = arr_mat[i].row_eff - kernel.row_eff + 1;
			int out_col_eff = arr_mat[i].col_eff - kernel.col_eff + 1;
			
			init_matrix(&out, out_row_eff, out_col_eff);

			omp_set_nested(1);

			#pragma omp parallel for num_threads(2) collapse(2)
			for (int j = 0; j < out.row_eff; j++) {
				for (int k = 0; k < out.col_eff; k++) {

					// START OF SUPPRESION
					// out.mat[i][j] = supression_op(kernel, target, i, j);
					int sumSupress = 0;
					int rowSupress = j;
					int colSupress = k;
					printf("Hello world from processor %s, rank %d out of %d processors, from thread %d out of %d threads\n", processor_name, world_rank, world_size, omp_get_thread_num(), omp_get_num_threads());
					

					#pragma omp parallel for num_threads(2) collapse(2) reduction(+:sumSupress)
					for (int l = 0; l < kernel.row_eff; l++) {
						for (int m = 0; m< kernel.col_eff; m++) {
							int nthreads, tid;
						
							nthreads = omp_get_num_threads();
							tid = omp_get_thread_num();
							printf("Hello world from [%d][%d] threadId %d out of %d threads\n", l,m, tid, nthreads);
							// printf("Kernel.mat[%d][%d]: %d dan arr_mat[%d].mat[%d + %d][%d + %d]:%d\n", l, m, kernel.mat[l][m],  i, rowSupress, l, colSupress, m, arr_mat[i].mat[rowSupress+l][colSupress+m]);
							sumSupress += kernel.mat[l][m] * arr_mat[i].mat[rowSupress + l][colSupress + m];
							// printf("Hasil sum: %d target[%d + %d][%d + %d]\n", sumSupress, rowSupress, l, colSupress, m);

						}
					}

					
					out.mat[j][k] = sumSupress;
					printf("Out.mat[%d][%d]: %d  SumSupress: %d\n",j, k, out.mat[j][k], sumSupress);
				}
			}
			arr_mat[i] = out;
			print_matrix(&arr_mat[i]);

			// END OF CONVOLUTION
			arr_range[i] = get_matrix_datarange(&arr_mat[i]); 
			MPI_Gather(&arr_range[i], 1, MPI_INT, arrRangeKonvo, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
		printf("BAJIGUR\n");
		// collects partial sums from other processes
		// int arr_range_total[num_targets];
		// int len_range_per_slave;
		// for (i = 1; i < world_size; i++) {
		// 	MPI_RECV(&len_range_per_slave, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// 	MPI_Recv(&arr_range_total, len_range_per_slave, MPI_INT,
		// 			MPI_ANY_SOURCE, 0,
		// 			MPI_COMM_WORLD,
		// 			&status);
		// 	int sender = status.MPI_SOURCE;
		// }

	}

	else {
		printf("WORLD RANK: %d", world_rank);
		// MPI_Recv(&sizeKernel, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);sss
		// Matrix kernel;
		// init_matrix(&kernel, 2, 2);
		MPI_Recv(&kernel, 10002, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&n_matrix_received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		arr_mat = (Matrix*)malloc(n_matrix_received * sizeof(Matrix));
		MPI_Recv(arr_mat, n_matrix_received * 10002, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("MASUK SINI: n_matrix_received: %d  kernel_size: %d\n", n_matrix_received, sizeKernel);
		int arr_range[n_matrix_received];
		for (int i = 0; i < n_matrix_received; i++) {
			// arr_mat[i] = input_matrix(target_row, target_col);
			// arr_mat[i] = convolution(&kernel, &arr_mat[i]);
			printf("CIHuy\n");
			print_matrix(&kernel);
			printf("TEST: row+eff: %d  coll_eff:%d", kernel.row_eff, kernel.col_eff);
			printf("CIHuy\n");
			printf("CIHuy\n");
			print_matrix(&arr_mat[i]);
			printf("CIHuy\n");
			// Start of Convolution
			Matrix out;
			int out_row_eff = arr_mat[i].row_eff - kernel.row_eff + 1;
			int out_col_eff = arr_mat[i].col_eff - kernel.col_eff + 1;
			
			init_matrix(&out, out_row_eff, out_col_eff);

			omp_set_nested(1);
			printf("Processor name: %s rank: %d out of %d processor\n", processor_name, world_rank, world_size);

			#pragma omp parallel for num_threads(2) collapse(2)
			for (int j = 0; j < out.row_eff; j++) {
				for (int k = 0; k < out.col_eff; k++) {

					// START OF SUPPRESION
					// out.mat[i][j] = supression_op(kernel, target, i, j);
					int sumSupress = 0;
					int rowSupress = j;
					int colSupress = k;
					printf("Hello world outerloop convolution from [%d][%d] threadId %d out of %d threads\n", j,k, omp_get_thread_num(), omp_get_num_threads());

					#pragma omp parallel for num_threads(2) collapse(2) reduction(+:sumSupress)
					for (int l = 0; l < kernel.row_eff; l++) {
						for (int m = 0; m< kernel.col_eff; m++) {
							int nthreads, tid;
						
							nthreads = omp_get_num_threads();
							tid = omp_get_thread_num();
							printf("Hello world from [%d][%d] threadId %d out of %d threads\n", l,m, tid, nthreads);
							// printf("Kernel.mat[%d][%d]: %d dan arr_mat[%d].mat[%d + %d][%d + %d]:%d\n", l, m, kernel.mat[l][m],  i, rowSupress, l, colSupress, m, arr_mat[i].mat[rowSupress+l][colSupress+m]);
							sumSupress += kernel.mat[l][m] * arr_mat[i].mat[rowSupress + l][colSupress + m];
							// printf("Hasil sum: %d target[%d + %d][%d + %d]\n", sumSupress, rowSupress, l, colSupress, m);

						}
					}

					
					out.mat[j][k] = sumSupress;
					printf("Out.mat[%d][%d]: %d  SumSupress: %d\n",j, k, out.mat[j][k], sumSupress);
				}
			}
			arr_mat[i] = out;
			// END OF CONVOLUTION
			printf("HASIL KONVOLUSI: \n");
			print_matrix(&arr_mat[i]);
			arr_range[i] = get_matrix_datarange(&arr_mat[i]);
			MPI_Gather(&arr_range[i], 1, MPI_INT, arrRangeKonvo, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}

		// MPI_Send(&arr_range, n_matrix_received, MPI_INT, 0, 0, MPI_COMM_WORLD);
		// MPI_Gather(&arr_range, n_matrix_received, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, 0,MPI_COMM_WORLD);
	}

	// MPI_Gather(&n_matrix_received, 1, MPI_INT, arrRangeKonvo, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
	if (world_rank == 0) {
        printf("Processor %d has data: \n", world_rank);
        for (int i=0; i<num_targets; i++)
            printf("Range Konvo:%d\n ", arrRangeKonvo[i]);
        printf("\n");
    }

	printf("MPI FINALIZE\n");

	MPI_Finalize();

	return 0;
}
