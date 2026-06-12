/**
  ******************************************************************************
  * @file    weights_mgmt.h
  * @author  STM32Cube AI Studio Code Generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

#ifndef __USER_INIT_H__
#define __USER_INIT_H__

#include "network.h"

#define STAI_NETWORK_WEIGHTS_TOTAL_SIZE_BYTES STAI_NETWORK_WEIGHTS_SIZE_BYTES

// uncomment the line and change address to load the weights to a specific address
// #define WEIGHTS_DEST_ADDRESS 0x0
// or allocate a weights buffer in RAM by uncommenting the following lines:
/*
#define WEIGHTS_DEST_ADDRESS (stai_ptr)network_weights;
extern uint8_t network_weights[STAI_NETWORK_WEIGHTS_TOTAL_SIZE_BYTES];
*/

extern const stai_network_details g_network_details;
extern uintptr_t _weights_addr[STAI_NETWORK_WEIGHTS_NUM + 1];

// uncomment the line and change address to set the weights from a specific address
// #define WEIGHTS_SOURCE_ADDRESS 0x0

stai_return_code user_stai_network_init(stai_network* network);


#endif // __USER_INIT_H__
