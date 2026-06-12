/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h7xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stm32h747i_discovery.h"
#include "stm32h747i_discovery_lcd.h"
#include "stm32_lcd.h"
/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */
#define LCD_FRAME_BUFFER    0xD0100000  /* 1MB into SDRAM, after AI activations */
/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define JOY_RIGHT_Pin GPIO_PIN_5
#define JOY_RIGHT_GPIO_Port GPIOK
#define JOY_LEFT_Pin GPIO_PIN_4
#define JOY_LEFT_GPIO_Port GPIOK
#define Audio_INT_Pin GPIO_PIN_15
#define Audio_INT_GPIO_Port GPIOJ
#define OSC32_OUT_Pin GPIO_PIN_15
#define OSC32_OUT_GPIO_Port GPIOC
#define OSC32_IN_Pin GPIO_PIN_14
#define OSC32_IN_GPIO_Port GPIOC
#define JOY_UP_Pin GPIO_PIN_6
#define JOY_UP_GPIO_Port GPIOK
#define JOY_DOWN_Pin GPIO_PIN_3
#define JOY_DOWN_GPIO_Port GPIOK
#define TOUCH_INT_Pin GPIO_PIN_7
#define TOUCH_INT_GPIO_Port GPIOK
#define BL_CTRL_Pin GPIO_PIN_12
#define BL_CTRL_GPIO_Port GPIOJ
#define STLINK_TX_Pin GPIO_PIN_10
#define STLINK_TX_GPIO_Port GPIOA
#define STLINK_RX_Pin GPIO_PIN_9
#define STLINK_RX_GPIO_Port GPIOA
#define uSD_Detect_Pin GPIO_PIN_8
#define uSD_Detect_GPIO_Port GPIOI
#define PMOD_12_Pin GPIO_PIN_13
#define PMOD_12_GPIO_Port GPIOJ
#define CEC_CK_MCO1_Pin GPIO_PIN_8
#define CEC_CK_MCO1_GPIO_Port GPIOA
#define LED1_Pin GPIO_PIN_12
#define LED1_GPIO_Port GPIOI
#define LED2_Pin GPIO_PIN_13
#define LED2_GPIO_Port GPIOI
#define LED3_Pin GPIO_PIN_14
#define LED3_GPIO_Port GPIOI
#define DSI_Reset_Pin GPIO_PIN_3
#define DSI_Reset_GPIO_Port GPIOG
#define JOY_SEL_Pin GPIO_PIN_2
#define JOY_SEL_GPIO_Port GPIOK
#define OSC_OUT_Pin GPIO_PIN_1
#define OSC_OUT_GPIO_Port GPIOH
#define OSC_IN_Pin GPIO_PIN_0
#define OSC_IN_GPIO_Port GPIOH
#define ARD_D7_Pin GPIO_PIN_0
#define ARD_D7_GPIO_Port GPIOJ
#define LED4_Pin GPIO_PIN_15
#define LED4_GPIO_Port GPIOI
#define OTG_HS_OverCurrent_Pin GPIO_PIN_1
#define OTG_HS_OverCurrent_GPIO_Port GPIOJ
#define ARD_D8_Pin GPIO_PIN_5
#define ARD_D8_GPIO_Port GPIOJ
#define PMOD_8_Pin GPIO_PIN_15
#define PMOD_8_GPIO_Port GPIOB
#define ARD_D2_Pin GPIO_PIN_3
#define ARD_D2_GPIO_Port GPIOJ
#define ARD_D4_Pin GPIO_PIN_4
#define ARD_D4_GPIO_Port GPIOJ
#define PMOD_9_Pin GPIO_PIN_14
#define PMOD_9_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
