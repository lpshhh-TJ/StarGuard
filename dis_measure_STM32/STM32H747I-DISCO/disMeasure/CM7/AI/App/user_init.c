/**
  ******************************************************************************
  * @file    weights_mgmt.c
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


#include "user_init.h"

#include <string.h>

uintptr_t _weights_addr[STAI_NETWORK_WEIGHTS_NUM + 1];

// uncomment the line and change address to load the weights to a specific address
/*
uint8_t network_weights[STAI_NETWORK_WEIGHTS_TOTAL_SIZE_BYTES];
#define WEIGHTS_DEST_ADDRESS (stai_ptr)network_weights;
*/


/**
* @brief Initialize the network weights by copying them to a specific address or by setting the weights buffers address in RAM
* @param network: pointer to the network context to initialize
* @return error code of the initialization process
* @note if WEIGHTS_SOURCE_ADDRESS is defined, the weights buffers are expected to be already
  located at the specified address, and the function will only update the network context with the weights buffers addresses.
  if WEIGHTS_DEST_ADDRESS is defined, the weights buffers will be copied to the specified address, and the network context 
  will be updated with the new weights buffers addresses.
  if both WEIGHTS_SOURCE_ADDRESS and WEIGHTS_DEST_ADDRESS are defined, the function will copy the weights 
  from the source address to the destination address, and update the network context with the new weights buffers addresses.
*/
STAI_API_ENTRY
stai_return_code user_stai_network_init(stai_network* network)
{
  // Initalize the network context
  stai_return_code stai_err = stai_network_init(network);
  if (stai_err != STAI_SUCCESS) {
    return stai_err;
  }
  
  #if defined(WEIGHTS_SOURCE_ADDRESS) || defined(WEIGHTS_DEST_ADDRESS)
  uint32_t weights_sizes[STAI_NETWORK_WEIGHTS_NUM] = STAI_NETWORK_WEIGHTS_SIZES;
  stai_ptr weights_buffers[STAI_NETWORK_WEIGHTS_NUM];
  stai_ptr new_weights_start_addr = 0x0;
  #endif

/**
* The following code is used to set the weights buffers addresses in the network context or to copy the weights buffers to a specific address.
* The behavior is driven by the definition of WEIGHTS_SOURCE_ADDRESS macro:
* - if WEIGHTS_SOURCE_ADDRESS is defined, the function will set the weights buffers addresses in the 
*   network context with the addresses calculated from the source address.
*/  
#ifdef WEIGHTS_SOURCE_ADDRESS
  stai_ptr weight_source_address_ptr = (stai_ptr)WEIGHTS_SOURCE_ADDRESS;
  new_weights_start_addr = weight_source_address_ptr;
  for (stai_size i = 0; i < STAI_NETWORK_WEIGHTS_NUM; i++) {  
    // update weight buffer address in the network context 
    weights_buffers[i] = new_weights_start_addr;
    // offset for the next weight buffer copy
    new_weights_start_addr += weights_sizes[i];
  }
  
  stai_err = stai_network_set_weights(network, weights_buffers, STAI_NETWORK_WEIGHTS_NUM);
#endif
 
/**
* The following code is used to set the weights buffers addresses in the network context or to copy the weights buffers to a specific address.
* The behavior is driven by the definition of WEIGHTS_DEST_ADDRESS macro:
* - if WEIGHTS_DEST_ADDRESS is defined, the function will copy the weights buffers from the addresses in the network 
*   context to the destination address, and update the network context with the new weights buffers addresses.
*/  
#ifdef WEIGHTS_DEST_ADDRESS // if defined, means copy the weights to the specific address  
  stai_ptr weight_dest_address_ptr =  (stai_ptr)WEIGHTS_DEST_ADDRESS;
  stai_size n_weights = 0;
  
   /* Copy weight buffers in the new location and update weight buffer addresses in the network context */
  new_weights_start_addr = weight_dest_address_ptr;
 
  stai_err = stai_network_get_weights(network, weights_buffers, &n_weights);
 

  /* Copy each weights in the new location */
  for (stai_size i = 0; i < n_weights; i++) {
    /* copy content of weight buffer in the new location */
    memcpy((void*)new_weights_start_addr, (const void*)weights_buffers[i], weights_sizes[i]);
    /* update weight buffer address in the network context */
    weights_buffers[i] = new_weights_start_addr;
    /* offset for the next weight buffer copy */
    new_weights_start_addr += weights_sizes[i];
  }
  stai_err = stai_network_set_weights(network, weights_buffers, n_weights);
#endif
  return stai_err;
}