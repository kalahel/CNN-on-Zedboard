#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "denseRelu.h"

#define SIZE 1000
#define NUMBER_OF_NEURON 20

void findDividersSimple(int inputSize, int divider[2]);

int main(int argc, char** argv) {
	volatile float A[SIZE] = { 0 };
	volatile float B[SIZE * NUMBER_OF_NEURON] = { 0 };
	volatile float res[NUMBER_OF_NEURON] = { 0 };

	int i;
	for (i = 0; i < SIZE; i++) {
		A[i] = (float) 1;
	}
	for (i = 0; i < SIZE * NUMBER_OF_NEURON; i++) {
		B[i] = (float) i * 0.001;

	}
	int baseaddr = 0;
	int divider[2] = { 0 };
	findDividersSimple(SIZE, divider);
	   clock_t t;
	    t = clock();
	denseRelu(A, res, B, divider[0], divider[1], NUMBER_OF_NEURON, baseaddr);
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds


	for (i = 0; i < NUMBER_OF_NEURON; i++) {
		printf("%d : %f => %f\r\n", i, B[i], res[i]);
	}
    printf("func() took %f seconds to execute \n", time_taken);

}


void findDividersSimple(int inputSize, int divider[2]) {

	for (int i = 255; i >= 0; i--) {
		if (inputSize % i == 0) {
			divider[0] = i;
			divider[1] = inputSize /i;
			break;
		}
	}
	printf("Highest divider for : %d\t[%d,%d]\n", inputSize, divider[0],
			divider[1]);
}

