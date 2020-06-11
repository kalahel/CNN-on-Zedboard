#ifndef DENSE_RELU_H_
#define DENSE_RELU_H_

void denseRelu(volatile float *inputs, volatile float *outputs,
		volatile float *weights, volatile float *biais,
		int inputSizeGreatestDivider, int inputSizeSmallestDivider,
		int numberOfNeurons, int baseaddr);
#endif
