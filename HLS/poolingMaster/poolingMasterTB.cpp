#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "poolingMaster.h"

int main(int argc, char** argv) {
	float data[6 * 4] = { 7, 5, 6, 3, 8, 2, 9, 1, 2, 4, 0, 9, 2, 4, 3, 1, 9, 5,
			5, 6, 0, 2, 6, 2 };
	float outputData[3*2] = {0};

	poolingMaster(data, outputData, 4, 6);

	for (int i = 0; i < 3*2; i++) {
		printf("%d : %f\r\n", i, outputData[i]);
	}

}

