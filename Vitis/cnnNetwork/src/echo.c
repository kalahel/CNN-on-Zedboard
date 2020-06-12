/*
 * Copyright (C) 2009 - 2019 Xilinx, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include <stdio.h>
#include <string.h>
#include "weights.h"
#include "xneuronv5relu.h"
#include "xconvmasterv2.h"
#include "xpoolingmaster.h"
#include "xtime_l.h"

#include "lwip/err.h"
#include "lwip/tcp.h"
#if defined (__arm__) || defined (__aarch64__)
#include "xil_printf.h"
#endif

#define VERBOSE	0
#define size_max_r 	100000000

char buffer_r[size_max_r];
long actual_position = 0;
int started = 0;
volatile float buffer_data[size_max_r];
int nb_elem_data;

void findDividersSimple(int inputSize, int divider[2]);

int transfer_data() {
	return 0;
}

void printReceivedData(float * data, int size) {
	xil_printf("---Received data---\r\n");
	for (int i = 0; i < size; ++i) {
		xil_printf("%d : %d => %f\r\n", i, data[i], data[i]);
	}
}
void consumDataDirty() {

	print("CNN Integration\n\r");
	int divider[2] = { 0 };

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
	findDividersSimple(13 * 13, divider);
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
	volatile float convOutput[26 * 26] = { 0 };
	volatile float poolOutput[13 * 13] = { 0 };
	volatile float dense1Out[64] = { 0 };
	volatile float dense2Out[32] = { 0 };
	volatile float dense3Out[10] = { 0 };

	XConvmasterv2_Set_outputs(&convip, convOutput);
	XPoolingmaster_Set_outputs(&poolip, poolOutput);
	XNeuronv5relu_Set_outputs(&xneuronIp, dense1Out);
	XNeuronv5relu_Set_outputs(&xneuron2Ip, dense2Out);
	XNeuronv5relu_Set_outputs(&xneuron3Ip, dense3Out);

	XConvmasterv2_Set_inputs(&convip, buffer_data);
	XPoolingmaster_Set_inputs(&poolip, convOutput);
	XNeuronv5relu_Set_inputs(&xneuronIp, poolOutput);
	XNeuronv5relu_Set_inputs(&xneuron2Ip, dense1Out);
	XNeuronv5relu_Set_inputs(&xneuron3Ip, dense2Out);

	XTime tstart, tend;
	XTime_GetTime(&tstart);

	XConvmasterv2_Start(&convip);
	while ((!XConvmasterv2_IsDone(&convip)) && !XConvmasterv2_IsIdle(&convip))
		;
	XPoolingmaster_Start(&poolip);
	while ((!XPoolingmaster_IsDone(&poolip)) && !XPoolingmaster_IsIdle(&poolip))
		;

	XNeuronv5relu_Start(&xneuronIp);
	while ((!XNeuronv5relu_IsReady(&xneuronIp))
			&& !XNeuronv5relu_IsIdle(&xneuronIp))
		;

	XNeuronv5relu_Start(&xneuron2Ip);
	while ((!XNeuronv5relu_IsReady(&xneuron2Ip))
			&& !XNeuronv5relu_IsIdle(&xneuron2Ip))
		;

	XNeuronv5relu_Start(&xneuron3Ip);
	while ((!XNeuronv5relu_IsReady(&xneuron3Ip))
			&& !XNeuronv5relu_IsIdle(&xneuron3Ip))
		;
	 XTime_GetTime(&tend);

	printf("___CNN PREDICTION___\r\n");
	printf("Output took %llu clock cycles.\n", (tend - tstart));
	printf("Output took %.2f us.\n", 1.0 * (tend - tstart) / (COUNTS_PER_SECOND/1000000));
	for (int i = 0; i < 10; ++i) {
		printf("%d : %.6f\r\n", i, dense3Out[i]);
	}

}

void print_app_header() {
#if (LWIP_IPV6==0)
	xil_printf("\n\r\n\r-----lwIP TCP echo server ------\n\r");
#else
	xil_printf("\n\r\n\r-----lwIPv6 TCP echo server ------\n\r");
#endif
	xil_printf("TCP packets sent to port 7 will be echoed back\n\r");

	xil_printf("TCP SND BUF = %d", TCP_SND_BUF);

}

int data_copy(struct pbuf *p) {
	char starting_word[] = "<start>";
	char ending_word[] = "<end>";
	char reset_word[] = "<reset>";

	if (actual_position + p->len < size_max_r) {
		if (p->len < 5 && started == 1) {
			memcpy((void*) &buffer_r[actual_position], p->payload, p->len);
			actual_position += p->len;
			buffer_r[actual_position] = '\0';
			return 1;
		}

		char extracted_starting_char[7];

		char extracted_ending_char[5];
		char extracted_ending_char2[5];

		char extracted_reset_char[7];

		// starting token
		memcpy(extracted_starting_char, p->payload, 7);
		extracted_starting_char[7] = '\0';

		//ending token
		memcpy(extracted_ending_char, &(p->payload)[p->len - 5], 5);
		memcpy(extracted_ending_char2, &(p->payload)[p->len - 6], 5);

		//reset token
		memcpy(extracted_reset_char, p->payload, 7);

		extracted_ending_char2[5] = '\0';
		extracted_ending_char[5] = '\0';
		extracted_reset_char[7] = '\0';
		if (VERBOSE) {
			xil_printf("Ending flag = $%s$\n\r", extracted_ending_char);
			xil_printf("Ending flag2 = $%s$\n\r", extracted_ending_char2);
			xil_printf("Reset flag = $%s$\n\r", extracted_reset_char);
		}
		// flag comparaison

		//start
		int start_flag = strcmp(extracted_starting_char, starting_word);
		//end
		int end_flag = strcmp(extracted_ending_char, ending_word);
		int end_flag2 = strcmp(extracted_ending_char2, ending_word);
		//reset
		int reset_flag = strcmp(extracted_reset_char, reset_word);

		if (VERBOSE) {
			xil_printf("\n\n\rTest Start %d \t End %d or %d\n\r", start_flag,
					end_flag, end_flag2);
			xil_printf("\n\n\rReset Flag %d\n\r", reset_flag);
		}

		if (reset_flag == 0) {
			started = 0;
			actual_position = 0;
			buffer_r[actual_position] = '\0';
			return 2;
		}

		if (start_flag == 0 && started == 0) {
			started = 1;
			actual_position = 0;
			memcpy((void*) &buffer_r[actual_position], &(p->payload)[7],
					p->len - 7); //<<<<<<<<<<<<<<
			actual_position += p->len - 7;
			buffer_r[actual_position] = '\0';

			if (end_flag == 0 || end_flag2 == 0) {
				if (end_flag == 0) {
					actual_position += -5;
				} else {
					actual_position += -6;
				}
				buffer_r[actual_position] = '\0';
				started = 0;
				return 0;

			} else {
				return 1;
			}

		} else if (started == 1) {

			memcpy((void*) &buffer_r[actual_position], p->payload, p->len);
			actual_position += p->len;
			buffer_r[actual_position] = '\0';
			if (end_flag == 0 || end_flag2 == 0) {
				if (end_flag == 0) {
					actual_position += -5;
				} else {
					actual_position += -6;
				}
				buffer_r[actual_position] = '\0';
				started = 0;
				return 0;

			} else {
				return 1;
			}
		} else {
			started = 0;
			actual_position = 0;
			buffer_r[actual_position] = '\0';
			xil_printf("INCORRECT STARTED\n\r");
			return -1;

		}

	} else {

		xil_printf(
				"################ INCORRECT MESSAGE ##########################\n\r");

		return -1;
	}

}

int data_processing() {
	int count = 0;
	int data_type;
	memcpy(&data_type, &(buffer_r)[0], 4 * sizeof(char));
	for (int i = 4; i < actual_position; i = i + 4) {
		float res;
		memcpy(&res, &(buffer_r)[i], 4 * sizeof(char));
		buffer_data[count] = res;
		count++;
	}
	nb_elem_data = count;

	return data_type;

}

err_t recv_callback(void *arg, struct tcp_pcb *tpcb, struct pbuf *p, err_t err) {
	/* do not read the packet if we are not in ESTABLISHED state */
	if (!p) {
		tcp_close(tpcb);
		tcp_recv(tpcb, NULL);
		return ERR_OK;
	}

	/* indicate that the packet has been received */
	tcp_recved(tpcb, p->len);
	int retour = -1;
	retour = data_copy(p);
	if (VERBOSE) {
		xil_printf("Return flag : %d\r\n", retour);
	}
	data_processing();
	if (VERBOSE) {
		xil_printf("Last [%d] = %d \n\r", nb_elem_data - 1,
				buffer_data[nb_elem_data - 1]);
	}

	if (nb_elem_data == 28 * 28 && retour == 0) {
		consumDataDirty();
	} else {
		if (VERBOSE) {
			xil_printf("Error size miss match : %d\r\n", nb_elem_data);
		}
	}
	if (tcp_sndbuf(tpcb) > actual_position) {
		if (retour == 0)
			err = tcp_write(tpcb, (void*) buffer_r, actual_position, 1);
	} else
		xil_printf("no space in tcp_sndbuf\n\r");

	/* free the received pbuf */
	pbuf_free(p);

	return ERR_OK;
}

err_t accept_callback(void *arg, struct tcp_pcb *newpcb, err_t err) {
	static int connection = 1;

	/* set the receive callback for this connection */
	tcp_recv(newpcb, recv_callback);

	/* just use an integer number indicating the connection id as the
	 callback argument */
	tcp_arg(newpcb, (void*) (UINTPTR) connection);

	/* increment for subsequent accepted connections */
	connection++;

	return ERR_OK;
}

int start_application() {
	struct tcp_pcb *pcb;
	err_t err;
	unsigned port = 7;

	/* create new TCP PCB structure */
	pcb = tcp_new_ip_type(IPADDR_TYPE_ANY);
	if (!pcb) {
		xil_printf("Error creating PCB. Out of Memory\n\r");
		return -1;
	}

	/* bind to specified @port */
	err = tcp_bind(pcb, IP_ANY_TYPE, port);
	if (err != ERR_OK) {
		xil_printf("Unable to bind to port %d: err = %d\n\r", port, err);
		return -2;
	}

	/* we do not need any arguments to callback functions */
	tcp_arg(pcb, NULL);

	/* listen for connections */
	pcb = tcp_listen(pcb);
	if (!pcb) {
		xil_printf("Out of memory while tcp_listen\n\r");
		return -3;
	}

	/* specify callback to use for incoming connections */
	tcp_accept(pcb, accept_callback);

	xil_printf("TCP echo server started @ port %d\n\r", port);

	return 0;
}

void findDividersSimple(int inputSize, int divider[2]) {

	for (int i = 255; i >= 0; i--) {
		if (inputSize % i == 0) {
			divider[0] = i;
			divider[1] = inputSize / i;
			break;
		}
	}
	if (VERBOSE) {
		printf("Highest divider for : %d\t[%d,%d]\n", inputSize, divider[0],
				divider[1]);
	}
}
