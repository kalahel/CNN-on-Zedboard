#include <stdlib.h>
#include <strings.h>
#include "denseRelu.h"

static float inputBuffer[256] = { 0 };
static float weightBuffer[256] = { 0 };
static float outputBuffer[100] = { 0 };
static float biaisBuffer[256] = { 0 };

// Warning this component is limited to layers of 256 neurons
void denseRelu(volatile float *inputs, volatile float *outputs,
		volatile float *weights, volatile float *biais,
		int inputSizeGreatestDivider, int inputSizeSmallestDivider,
		int numberOfNeurons, int baseaddr) {

#pragma HLS INTERFACE m_axi depth=51200 port=inputs offset=slave bundle=AXI_INPUTS max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=100 port=outputs offset=slave bundle=AXI_OUTPUT
#pragma HLS INTERFACE m_axi depth=5120000 port=weights offset=slave bundle=AXI_WEIGHT max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=200 port=biais offset=slave bundle=AXI_BIAIS

#pragma HLS INTERFACE s_axilite port=inputs bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=outputs bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=weights bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=biais bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=baseaddr bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=inputSizeGreatestDivider bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=inputSizeSmallestDivider bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=numberOfNeurons bundle=CONTROL_BUS
#pragma HLS INTERFACE ap_stable port=baseaddr bundle=CONTROL_BUS
#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS

	memcpy(biaisBuffer, (float*) biais, numberOfNeurons * sizeof(float));

	neuronV5_label0: for (int wi = 0; wi < numberOfNeurons; wi++) {
		outputBuffer[wi] = 0;
		// Tying to prevent IP stalling caused by overflowing the burst buffer size
		neuronV5_label1: for (int i = 0; i < inputSizeSmallestDivider; i++) {
			memcpy(inputBuffer,
					(float*) (inputs + i * inputSizeGreatestDivider),
					inputSizeGreatestDivider * sizeof(float));
			memcpy(weightBuffer,
					(float*) (weights + i * inputSizeGreatestDivider
							+ wi * inputSizeGreatestDivider
									* inputSizeSmallestDivider),
					inputSizeGreatestDivider * sizeof(float));
			//Compute output
			neuronV5_label2: for (int j = 0; j < inputSizeGreatestDivider;
					j++) {
				outputBuffer[wi] += inputBuffer[j] * weightBuffer[j];
			}

		}
		outputBuffer[wi] += biaisBuffer[wi];
		// Relu application
		if (outputBuffer[wi] < 0.0)
			outputBuffer[wi] = 0.0;
	}

	memcpy((float*) (outputs), outputBuffer, numberOfNeurons * sizeof(float));

}
