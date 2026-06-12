/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    stm32h7xx_it.c
  * @brief   Interrupt Service Routines.
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
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "stm32h7xx_it.h"
/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
extern UART_HandleTypeDef huart1;
#include "stm32_lcd.h"
#include "fonts.h"
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN TD */

/* USER CODE END TD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/* External variables --------------------------------------------------------*/

/* USER CODE BEGIN EV */

/* USER CODE END EV */

/******************************************************************************/
/*           Cortex Processor Interruption and Exception Handlers          */
/******************************************************************************/
/**
  * @brief This function handles Non maskable interrupt.
  */
void NMI_Handler(void)
{
  /* USER CODE BEGIN NonMaskableInt_IRQn 0 */

  /* USER CODE END NonMaskableInt_IRQn 0 */
  /* USER CODE BEGIN NonMaskableInt_IRQn 1 */
   while (1)
  {
  }
  /* USER CODE END NonMaskableInt_IRQn 1 */
}

/**
  * @brief This function handles Hard fault interrupt.
  */
void HardFault_Handler(void)
{
  /* USER CODE BEGIN HardFault_IRQn 0 */

  /* USER CODE END HardFault_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_HardFault_IRQn 0 */
    /* USER CODE END W1_HardFault_IRQn 0 */
  }
}

/**
  * @brief This function handles Memory management fault.
  */
void MemManage_Handler(void)
{
  /* USER CODE BEGIN MemoryManagement_IRQn 0 */

  /* USER CODE END MemoryManagement_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_MemoryManagement_IRQn 0 */
    /* USER CODE END W1_MemoryManagement_IRQn 0 */
  }
}

/**
  * @brief This function handles Pre-fetch fault, memory access fault.
  */
void BusFault_Handler(void)
{
  /* USER CODE BEGIN BusFault_IRQn 0 */

  /* USER CODE END BusFault_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_BusFault_IRQn 0 */
    /* USER CODE END W1_BusFault_IRQn 0 */
  }
}

/**
  * @brief This function handles Undefined instruction or illegal state.
  */
void UsageFault_Handler(void)
{
  /* USER CODE BEGIN UsageFault_IRQn 0 */

  /* USER CODE END UsageFault_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_UsageFault_IRQn 0 */
    /* USER CODE END W1_UsageFault_IRQn 0 */
  }
}

/**
  * @brief This function handles System service call via SWI instruction.
  */
void SVC_Handler(void)
{
  /* USER CODE BEGIN SVCall_IRQn 0 */

  /* USER CODE END SVCall_IRQn 0 */
  /* USER CODE BEGIN SVCall_IRQn 1 */

  /* USER CODE END SVCall_IRQn 1 */
}

/**
  * @brief This function handles Debug monitor.
  */
void DebugMon_Handler(void)
{
  /* USER CODE BEGIN DebugMonitor_IRQn 0 */

  /* USER CODE END DebugMonitor_IRQn 0 */
  /* USER CODE BEGIN DebugMonitor_IRQn 1 */

  /* USER CODE END DebugMonitor_IRQn 1 */
}

/**
  * @brief This function handles Pendable request for system service.
  */
void PendSV_Handler(void)
{
  /* USER CODE BEGIN PendSV_IRQn 0 */

  /* USER CODE END PendSV_IRQn 0 */
  /* USER CODE BEGIN PendSV_IRQn 1 */

  /* USER CODE END PendSV_IRQn 1 */
}

/**
  * @brief This function handles System tick timer.
  */
void SysTick_Handler(void)
{
  /* USER CODE BEGIN SysTick_IRQn 0 */

  /* USER CODE END SysTick_IRQn 0 */
  HAL_IncTick();
  /* USER CODE BEGIN SysTick_IRQn 1 */

  /* USER CODE END SysTick_IRQn 1 */
}

/******************************************************************************/
/* STM32H7xx Peripheral Interrupt Handlers                                    */
/* Add here the Interrupt Handlers for the used peripherals.                  */
/* For the available peripheral interrupt handler names,                      */
/* please refer to the startup file (startup_stm32h7xx.s).                    */
/******************************************************************************/

/**
  * @brief This function handles TIM6 global interrupt (LCD periodic refresh).
  */
void TIM6_DAC_IRQHandler(void)
{
  if(TIM6->SR & TIM_SR_UIF)
  {
    TIM6->SR = ~TIM_SR_UIF;

    extern volatile int stat_bytes, stat_frames_l1, stat_frames_l2;
    extern volatile int stat_pairs, stat_crc_err, stat_dropped;
    void uart_parse_frames(void);

    /* Parse any pending binary frames in the ring buffer */
    uart_parse_frames();

    /* Update LCD with receiver statistics — only on change to reduce flicker */
    {
      static int prev_bytes, prev_l1, prev_l2, prev_pairs, prev_crc, prev_drop;
      if (stat_bytes != prev_bytes || stat_frames_l1 != prev_l1 || stat_frames_l2 != prev_l2 || stat_pairs != prev_pairs || stat_crc_err != prev_crc || stat_dropped != prev_drop) {
        prev_bytes  = stat_bytes;
        prev_l1     = stat_frames_l1;
        prev_l2     = stat_frames_l2;
        prev_pairs  = stat_pairs;
        prev_crc    = stat_crc_err;
        prev_drop   = stat_dropped;

        char line[40];
        UTIL_LCD_SetTextColor(0xFFCCAA44);
        UTIL_LCD_DisplayStringAt(430, LINE(15), (uint8_t *)"UART RX Stats", LEFT_MODE);
        UTIL_LCD_SetTextColor(0xFFCCCCCC);
        snprintf(line, sizeof(line), "Bytes:%d L1:%d L2:%d", stat_bytes, stat_frames_l1, stat_frames_l2);
        UTIL_LCD_DisplayStringAt(430, LINE(16), (uint8_t *)line, LEFT_MODE);
        snprintf(line, sizeof(line), "Pairs:%d CRCerr:%d", stat_pairs, stat_crc_err);
        UTIL_LCD_DisplayStringAt(430, LINE(17), (uint8_t *)line, LEFT_MODE);
        snprintf(line, sizeof(line), "Dropped:%d", stat_dropped);
        UTIL_LCD_DisplayStringAt(430, LINE(18), (uint8_t *)line, LEFT_MODE);
      }
    }
  }
}

/* USER CODE BEGIN 1 */

/* Extern declarations for binary frame receiver (UART8 -> ring buffer) */
extern char rx_buf[4096];
extern volatile uint16_t rx_wp;

/* UART8 receives IQ data frames from external sensor (Arduino D0/D1).
   Custom byte-at-a-time ISR writes directly to ring buffer for low latency. */
void UART8_IRQHandler(void)
{
  uint32_t isr = UART8->ISR;
  if(isr & (USART_ISR_ORE | USART_ISR_FE | USART_ISR_PE))
  {
    volatile uint32_t dummy = UART8->RDR;
    UART8->ICR = USART_ICR_ORECF | USART_ICR_FECF | USART_ICR_PECF;
    (void)dummy;
  }

  if(isr & USART_ISR_RXNE_RXFNE)
  {
    uint8_t ch = (uint8_t)(UART8->RDR & 0xFF);
    rx_buf[rx_wp] = (char)ch;
    rx_wp = (rx_wp + 1) % 4096;
  }
}

/* USART1 (STLink VCP) — used only for debug logging (LC_PRINT / HAL_UART_Transmit).
   Interrupt handled by HAL for proper error recovery. */
extern UART_HandleTypeDef huart1;

void USART1_IRQHandler(void)
{
  HAL_UART_IRQHandler(&huart1);
}

/* USER CODE END 1 */
