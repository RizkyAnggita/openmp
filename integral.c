#include "omp.h"
#include <stdio.h>
#define NUM_THREADS 4

static long num_steps = 1000;
double step;


int main() {
    double sumTotal[NUM_THREADS];
    double timeStart, timeEnd;
    double pi = 0.0;
    int i;
    omp_set_num_threads(NUM_THREADS);
    step = 1.0/(double) num_steps;
    timeStart = omp_get_wtime();

    #pragma omp parallel 
    {
        int i, tid, start, end, nthreads;
        double x;
        double sum = 0.0;
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        printf("NUMTHREAD: %d\n", nthreads);
        
        // start = (num_steps/nthreads)*tid;
        // end = start + (num_steps/nthreads);
        // // result[tid] = 0.0;
        
        // for (i = start, sum[tid]=0.0; i < end; i++)
        // {
        //     x = (i+0.5)*step;
        //     sum[tid] += 4.0/(1.0 + x*x);

        // }
        
        for (i = tid; i < num_steps; i+=nthreads)
        {
            x = (i+0.5)*step;
            sum += 4.0/(1.0 + x*x);

        }

        #pragma omp critical
            pi += sum * step;

    }
    
    timeEnd = omp_get_wtime();
    printf("Result: %.8f\n", pi);
    printf("Time: %.8f\n", (timeEnd-timeStart) * 1000);
}