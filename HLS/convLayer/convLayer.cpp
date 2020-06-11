#include <stdlib.h>
#include <strings.h>
#include "convLayer.h"



void convLayer(volatile float *inputs, volatile float *outputs,
		volatile float *filter, int rows, int cols) {

#pragma HLS INTERFACE m_axi depth=2073600 port=inputs offset=slave bundle=AXI_INPUTS
#pragma HLS INTERFACE m_axi depth=2073600 port=outputs offset=slave bundle=AXI_OUTPUT
#pragma HLS INTERFACE m_axi depth=9 port=filter offset=slave bundle=AXI_OUTPUT

#pragma HLS INTERFACE s_axilite port=inputs bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=outputs bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=filter bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=rows bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=cols bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS

	float window[9] = { 0 };
	float filterBuffer[9] = { 0 };

	memcpy(filterBuffer, (float*)filter, 9 * sizeof(float));
	int outputIndex = 0;

	poolingMaster_label0: for (int i = 0; i < rows - 2; i += 1) {
		poolingMaster_label1: for (int j = 0; j < cols - 2; j += 1) {
			float acc = 0;
			// upper part of the window
			memcpy(window, (float *) (inputs + j + (i * cols)),
					3 * sizeof(float));
			// middle part
			memcpy((float *) (window + 3),
					(float *) (inputs + j + ((i + 1) * cols)),
					3 * sizeof(float));
			// lower part
			memcpy((float *) (window + 6),
					(float *) (inputs + j + ((i + 2) * cols)),
					3 * sizeof(float));

			for (int k = 0; k < 9; ++k) {
#pragma HLS LOOP_FLATTEN
				acc += window[k] * filterBuffer[k];
			}
			float outputBuffer[1];
			outputBuffer[0] = acc;
			memcpy((float *) (outputs + outputIndex), outputBuffer,
					sizeof(float));
			outputIndex++;

		}
	}

}
