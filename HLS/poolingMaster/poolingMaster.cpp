
#include <stdlib.h>
#include <strings.h>
#include "poolingMaster.h"

static float window[4] = { 0 };

void poolingMaster(volatile float *inputs, volatile float *outputs, int rows,
		int cols) {

#pragma HLS INTERFACE m_axi depth=2073600 port=inputs offset=slave bundle=AXI_INPUTS
#pragma HLS INTERFACE m_axi depth=2073600 port=outputs offset=slave bundle=AXI_OUTPUT

#pragma HLS INTERFACE s_axilite port=inputs bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=outputs bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=rows bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=cols bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS

	int outputIndex = 0;
	poolingMaster_label0: for (int i = 0; i < rows - 1; i += 2) {
		poolingMaster_label1: for (int j = 0; j < cols - 1; j += 2) {
			// upper part of the window
			memcpy(window, (float *) (inputs + j + (i * cols)),
					2 * sizeof(float));
			memcpy((float *) (window + 2),
					(float *) (inputs + j + ((i + 1) * cols)),
					2 * sizeof(float));
			float max = -9999.0;
			for (int k = 0; k < 4; ++k) {
#pragma HLS LOOP_FLATTEN
				if (window[k] > max) {
					max = window[k];
				}
			}
			float outputBuffer[1];
			outputBuffer[0] = max;
			memcpy((float *) (outputs + outputIndex), outputBuffer,
					sizeof(float));
			outputIndex++;

		}
	}

}
