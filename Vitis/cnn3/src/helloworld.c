/******************************************************************************
 *
 * Copyright (C) 2009 - 2014 Xilinx, Inc.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * Use of the Software is limited solely to applications:
 * (a) running on a Xilinx device, or
 * (b) that interact with a Xilinx device through a bus or interconnect.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
 * OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Except as contained in this notice, the name of the Xilinx shall not be used
 * in advertising or otherwise to promote the sale, use or other dealings in
 * this Software without prior written authorization from Xilinx.
 *
 ******************************************************************************/

/*
 * helloworld.c: simple test application
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */

#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xneuronv5relu.h"
#include "xconvmasterv2.h"
#include "xpoolingmaster.h"
#include "weights.h"

void findDividersSimple(int inputSize, int divider[2]);



int main() {
	init_platform();
	Xil_DCacheDisable();

	print("CNN Integration\n\r");

	int divider[2] = {0};
	// Conv 3*3
	XConvmasterv2 convip;
	XConvmasterv2_Config * convConfig = XConvmasterv2_LookupConfig(
	XPAR_XCONVMASTERV2_0_DEVICE_ID);
	int status = XConvmasterv2_CfgInitialize(&convip, convConfig);
	if (status != 0)
		printf("ERROR\n");
	status = XConvmasterv2_Initialize(&convip,
			XPAR_XCONVMASTERV2_0_DEVICE_ID);
	if (status != 0)
		printf("ERROR\n");
	XConvmasterv2_Set_filter(&convip, convKernel());
	XConvmasterv2_Set_cols(&convip, 28);
	XConvmasterv2_Set_rows(&convip, 28);
	//XConvmasterv2_EnableAutoRestart(&convip);

	// Pooling
	XPoolingmaster poolip;
	XPoolingmaster_Config * poolConfig = XPoolingmaster_LookupConfig(
	XPAR_XCONVMASTERV2_0_DEVICE_ID);
	status = XPoolingmaster_CfgInitialize(&poolip, poolConfig);
	if (status != 0)
		printf("ERROR\n");
	status = XPoolingmaster_Initialize(&poolip,
			XPAR_XCONVMASTERV2_0_DEVICE_ID);
	if (status != 0)
		printf("ERROR\n");
	XPoolingmaster_Set_cols(&poolip, 26);
	XPoolingmaster_Set_rows(&poolip, 26);
	//XPoolingmaster_EnableAutoRestart(&poolip);

	// Dense 1 - 64
	XNeuronv5relu xneuronIp;
	XNeuronv5relu_Config * xneuronConfig = XNeuronv5relu_LookupConfig(
	XPAR_XNEURONV5RELU_0_DEVICE_ID);
	status = XNeuronv5relu_CfgInitialize(&xneuronIp, xneuronConfig);
	if (status != 0)
		printf("ERROR\n");
	status = XNeuronv5relu_Initialize(&xneuronIp,
	XPAR_XNEURONV5RELU_0_DEVICE_ID);
	if (status != 0)
		printf("ERROR\n");
	XNeuronv5relu_DisableAutoRestart(&xneuronIp);
	findDividersSimple(13*13, divider);
	XNeuronv5relu_Set_numberOfNeurons(&xneuronIp, 64);
	XNeuronv5relu_Set_inputSizeGreatestDivider(&xneuronIp, divider[0]);
	XNeuronv5relu_Set_inputSizeSmallestDivider(&xneuronIp, divider[1]);
	XNeuronv5relu_Set_weights(&xneuronIp, layerDense1());
	XNeuronv5relu_Set_biais(&xneuronIp, biaisDense1());
	//XNeuronv5relu_EnableAutoRestart(&xneuronIp);


	// Dense 2 - 32
	XNeuronv5relu xneuron2Ip;
	XNeuronv5relu_Config * xneuron2Config = XNeuronv5relu_LookupConfig(
	XPAR_XNEURONV5RELU_1_DEVICE_ID);
	status = XNeuronv5relu_CfgInitialize(&xneuron2Ip, xneuron2Config);
	if (status != 0)
		printf("ERROR\n");
	status = XNeuronv5relu_Initialize(&xneuron2Ip,
	XPAR_XNEURONV5RELU_1_DEVICE_ID);
	if (status != 0)
		printf("ERROR\n");
	XNeuronv5relu_DisableAutoRestart(&xneuron2Ip);

	findDividersSimple(64, divider);
	XNeuronv5relu_Set_numberOfNeurons(&xneuron2Ip, 64);
	XNeuronv5relu_Set_inputSizeGreatestDivider(&xneuron2Ip, divider[0]);
	XNeuronv5relu_Set_inputSizeSmallestDivider(&xneuron2Ip, divider[1]);
	XNeuronv5relu_Set_weights(&xneuron2Ip, layerDense2());
	XNeuronv5relu_Set_biais(&xneuron2Ip, biaisDense2());
	//XNeuronv5relu_EnableAutoRestart(&xneuron2Ip);

	// Dense 3 - 10
	XNeuronv5relu xneuron3Ip;
	XNeuronv5relu_Config * xneuron3Config = XNeuronv5relu_LookupConfig(
	XPAR_XNEURONV5RELU_2_DEVICE_ID);
	status = XNeuronv5relu_CfgInitialize(&xneuron3Ip, xneuron3Config);
	if (status != 0)
		printf("ERROR\n");
	status = XNeuronv5relu_Initialize(&xneuron3Ip,
	XPAR_XNEURONV5RELU_2_DEVICE_ID);
	if (status != 0)
		printf("ERROR\n");
	XNeuronv5relu_DisableAutoRestart(&xneuron3Ip);
	findDividersSimple(32, divider);
	XNeuronv5relu_Set_numberOfNeurons(&xneuron3Ip, 32);
	XNeuronv5relu_Set_inputSizeGreatestDivider(&xneuron3Ip, divider[0]);
	XNeuronv5relu_Set_inputSizeSmallestDivider(&xneuron3Ip, divider[1]);
	XNeuronv5relu_Set_weights(&xneuron3Ip, layerDense3());
	XNeuronv5relu_Set_biais(&xneuron3Ip, biaisDense3());
	//XNeuronv5relu_EnableAutoRestart(&xneuron3Ip);



	// Ptr definition
	volatile float convOutput[26*26] = {0};
	volatile float poolOutput[13*13] = {0};
	volatile float dense1Out[64] = {0};
	volatile float dense2Out[32] = {0};
	volatile float dense3Out[10] = {0};


	XConvmasterv2_Set_outputs(&convip, convOutput);
	XPoolingmaster_Set_outputs(&poolip, poolOutput);
	XNeuronv5relu_Set_outputs(&xneuronIp, dense1Out);
	XNeuronv5relu_Set_outputs(&xneuron2Ip, dense2Out);
	XNeuronv5relu_Set_outputs(&xneuron3Ip, dense3Out);

	XConvmasterv2_Set_inputs(&convip, dataInput3());
	XPoolingmaster_Set_inputs(&poolip, convOutput);
	XNeuronv5relu_Set_inputs(&xneuronIp, poolOutput);
	XNeuronv5relu_Set_inputs(&xneuron2Ip, dense1Out);
	XNeuronv5relu_Set_inputs(&xneuron3Ip, dense2Out);




	XConvmasterv2_Start(&convip);
	while((!XConvmasterv2_IsDone(&convip)) && !XConvmasterv2_IsIdle(&convip));
	XPoolingmaster_Start(&poolip);
	while((!XPoolingmaster_IsDone(&poolip)) && !XPoolingmaster_IsIdle(&poolip));

	XNeuronv5relu_Start(&xneuronIp);
	while((!XNeuronv5relu_IsReady(&xneuronIp)) && ! XNeuronv5relu_IsIdle(&xneuronIp));

	XNeuronv5relu_Start(&xneuron2Ip);
	while((!XNeuronv5relu_IsReady(&xneuron2Ip)) && ! XNeuronv5relu_IsIdle(&xneuron2Ip));

	XNeuronv5relu_Start(&xneuron3Ip);
	while((!XNeuronv5relu_IsReady(&xneuron3Ip)) && ! XNeuronv5relu_IsIdle(&xneuron3Ip));


	/*
	XConvmasterv2_EnableAutoRestart(&convip);
	XPoolingmaster_EnableAutoRestart(&poolip);
	XNeuronv5relu_EnableAutoRestart(&xneuronIp);
	XNeuronv5relu_EnableAutoRestart(&xneuron2Ip);
	XNeuronv5relu_EnableAutoRestart(&xneuron3Ip);
	*/

	for (int i = 0; i < 10; ++i) {
		//printf("%d : %f\n", i, dense1Out[i]);
		printf("%d : %.6f\r\n", i, dense3Out[i]);
	}




	cleanup_platform();
	return 0;
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




