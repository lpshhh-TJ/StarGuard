
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
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

  /**
    * Description
    * v1.0: Minimum template to show how to use the Embedded Client API ST-AI 
    *
        */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#define AI_DTCMRAM   _Pragma("location=\"AI_DTCMRAM\"")
#define AI_ITCMRAM   _Pragma("location=\"AI_ITCMRAM\"")
#define AI_RAM_D1   _Pragma("location=\"AI_RAM_D1\"")
#define AI_FMC   _Pragma("location=\"AI_FMC\"")
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#define AI_DTCMRAM   __attribute__((section(".AI_DTCMRAM")))
#define AI_ITCMRAM   __attribute__((section(".AI_ITCMRAM")))
#define AI_RAM_D1   __attribute__((section(".AI_RAM_D1")))
#define AI_FMC   __attribute__((section(".AI_FMC")))
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include "app_x-cube-ai.h"
#include "bsp_ai.h"
#include "stai.h"



/* USER CODE BEGIN includes */
#include "main.h"
#include "fonts.h"
#include <math.h>
#include "multi_test.h"   /* 3 real training samples with labels */

extern UART_HandleTypeDef huart1;
UART_HandleTypeDef huart8;  /* UART8 (Arduino D0/D1) — IQ data input */

/* ── Normalization parameters (from distance.json) ───────────────────── */
#define INPUT_SCALE  8380418.0f
#define LABEL_SCALE  1.0f
#define DECODE_EPS   1e-3f

/* Input normalization: raw Hankel → [0,1] by dividing by max scale.
   Both real (ch0) and imaginary (ch1) use the same scale. */
static void normalize_hankel(float *inp)
{
  for (int i = 0; i < 3200; i++)
    inp[i] /= INPUT_SCALE;
}

/* Output denormalization: log-space model output → actual distance in meters.
   distance = exp(output * label_scale) - eps, clamped to [0, 10]. */
static float denormalize_output(float raw)
{
  float log_pred = raw * LABEL_SCALE;
  float d = expf(log_pred) - DECODE_EPS;
  if (d < 0.0f)   d = 0.0f;
  if (d > 10.0f)  d = 10.0f;
  return d;
}
/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/


/* Input defs ----------------------------------------------------------------*/
#include "aiTestUtility.h"
STAI_ALIGNED(32) static uint8_t data_in_1[STAI_NETWORK_IN_1_SIZE_BYTES];

// Array to store the data of the input tensor
stai_ptr data_ins[] = {
  data_in_1
};

/* Output defs ----------------------------------------------------------------*/

STAI_ALIGNED(32)
static uint8_t data_out_1[STAI_NETWORK_OUT_1_SIZE_BYTES];

// c-array to store the data of the output tensor
stai_ptr data_outs[] = {
  data_out_1
};




/* Global byte buffer to save instantiated C-model network context */
STAI_NETWORK_CONTEXT_DECLARE(network_context, STAI_NETWORK_CONTEXT_SIZE)

/* Activations buffers -------------------------------------------------------*/
STAI_ALIGNED(32) 
AI_DTCMRAM 
static uint8_t DTCMRAM[STAI_NETWORK_ACTIVATION_1_SIZE_BYTES];
STAI_ALIGNED(32) 
AI_RAM_D1 
static uint8_t RAM_D1[STAI_NETWORK_ACTIVATION_2_SIZE_BYTES];


/* Global c-array to handle the activations buffer */
stai_ptr data_activations[] = { DTCMRAM,RAM_D1,(ai_handle)0xd0000000 };

STAI_ALIGNED(32) static uint8_t states_1[4];
stai_ptr data_states[] = {
    states_1
};




/* Entry points --------------------------------------------------------------*/

/* Array of pointer to manage the model's input/output tensors */
static stai_size in_length, out_length;
static stai_ptr stai_input[STAI_NETWORK_IN_NUM];
static stai_ptr stai_output[STAI_NETWORK_OUT_NUM];


/* 
 * Bootstrap
 */
int aiInit(void) {
  stai_return_code ret_code;

  /* 1: Initialize runtime library */
  ret_code = stai_runtime_init();
  if (ret_code != STAI_SUCCESS) {
    LC_PRINT("aiInit: stai_runtime_init failed: %s\r\n", stai_get_return_code_name(ret_code));
    return -1;
  }

  /* 2: Initialize network model context */
  ret_code = user_stai_network_init(network_context);
  if (ret_code != STAI_SUCCESS) {
    LC_PRINT("aiInit: user_stai_network_init failed: %s\r\n", stai_get_return_code_name(ret_code));
    return -2;
  }
  /* 3: Set network activations buffers */
  ret_code = stai_network_set_activations(network_context, data_activations, STAI_NETWORK_ACTIVATIONS_NUM);
  if (ret_code != STAI_SUCCESS) {
    LC_PRINT("aiInit: set_activations failed: %s\r\n", stai_get_return_code_name(ret_code));
    return -3;
  }

  /* 4: Manually assign our own input/output buffers.
     The model uses allocate-inputs/outputs by default, but we override
     with explicit buffers so we control exactly where data is placed. */
  ret_code = stai_network_set_inputs(network_context, data_ins, STAI_NETWORK_IN_NUM);
  if (ret_code != STAI_SUCCESS) {
    LC_PRINT("aiInit: set_inputs failed: %s\r\n", stai_get_return_code_name(ret_code));
    return -4;
  }
  ret_code = stai_network_set_outputs(network_context, data_outs, STAI_NETWORK_OUT_NUM);
  if (ret_code != STAI_SUCCESS) {
    LC_PRINT("aiInit: set_outputs failed: %s\r\n", stai_get_return_code_name(ret_code));
    return -5;
  }

  /* Store convenience pointers (used by selftest & acquire) */
  stai_input[0]  = (stai_ptr)data_in_1;
  stai_output[0] = (stai_ptr)data_out_1;
  in_length  = STAI_NETWORK_IN_1_SIZE;
  out_length = STAI_NETWORK_OUT_1_SIZE;

  LC_PRINT("aiInit OK: own bufs in=%p (%u B), out=%p (%u B)\r\n",
           (void*)stai_input[0], (unsigned)in_length,
           (void*)stai_output[0], (unsigned)out_length);
  return 0;
}

int aiDeinit(void) {
  stai_return_code ret_code;

  /* 1: Deinitialize network model context */
  ret_code = stai_network_deinit(network_context);

  /* 2: Deinitialize runtime library */
  ret_code = stai_runtime_deinit();

  return 0;
}

/* ── Copy AI weights from QSPI to SDRAM for fast inference ───────────── */
#define WEIGHTS_SDRAM_BASE  0xD0200000U
#define WEIGHTS_TOTAL_SIZE  4912516U   /* 4.9 MB from network.h */

static int copy_weights_to_sdram(void)
{
  stai_ptr  src_bufs[STAI_NETWORK_WEIGHTS_NUM];
  stai_size n_w = 0;
  stai_return_code ret;

  ret = stai_network_get_weights(network_context, src_bufs, &n_w);
  if (ret != STAI_SUCCESS || n_w == 0) {
    LC_PRINT("copy_weights: get_weights failed\r\n");
    return -1;
  }

  uint32_t sizes[STAI_NETWORK_WEIGHTS_NUM] = STAI_NETWORK_WEIGHTS_SIZES;
  stai_ptr  dst_bufs[STAI_NETWORK_WEIGHTS_NUM];
  uint8_t  *dst_base = (uint8_t *)WEIGHTS_SDRAM_BASE;
  uint8_t  *dst = dst_base;
  uint32_t  total = 0;

  for (stai_size i = 0; i < n_w; i++) {
    uint8_t *src = (uint8_t *)src_bufs[i];
    uint32_t sz  = sizes[i];

    /* Use memcpy — the ARM Compiler 5 memcpy is highly optimized */
    memcpy(dst, src, sz);

    /* Verify at 5 sample points: start, 25%, 50%, 75%, end */
    uint32_t offs[5] = {0, sz/4, sz/2, (3*sz)/4, sz - 16};
    int ok = 1;
    for (int k = 0; k < 5; k++) {
      uint32_t o = offs[k] & ~3U;  /* word-align */
      if (o < sz && *(uint32_t*)(dst + o) != *(uint32_t*)(src + o)) {
        LC_PRINT("  FAIL @%lu: src=%08lx dst=%08lx\r\n",
                 (unsigned long)o,
                 (unsigned long)*(uint32_t*)(src+o),
                 (unsigned long)*(uint32_t*)(dst+o));
        ok = 0;
        break;
      }
    }
    if (!ok) return -2;

    dst_bufs[i] = (stai_ptr)dst;
    dst += sz;
    total += sz;
  }

  /* Final verification: SDRAM content must match QSPI at random offset */
  if (*(uint32_t*)(dst_base + total/2) != *(uint32_t*)((uint8_t*)src_bufs[0] + total/2))
    return -3;

  ret = stai_network_set_weights(network_context, dst_bufs, n_w);
  if (ret != STAI_SUCCESS) {
    LC_PRINT("copy_weights: set_weights failed: %s\r\n",
             stai_get_return_code_name(ret));
    return -4;
  }

  LC_PRINT("copy_weights: OK (%lu bytes to SDRAM)\r\n", (unsigned long)total);
  return 0;
}

/* 
 * Run inference
 */
stai_return_code aiRun() {
  stai_return_code ret_code;

  /** Profiling code to calculate the inference time of the model. You can remove it if not needed */
  static uint32_t inference_nb = 0;
  static uint32_t total_cycles = 0;
  uint32_t start_tick, end_tick, end_dwt = 0;
  struct dwtTime t;
  cyclesCounterInit();

  LC_PRINT("---- Inference number %" PRIu32 " ----\r\n", inference_nb);
  LC_PRINT("Results for network \"%s\"\r\nRunning...\r\n", STAI_NETWORK_MODEL_NAME);
  cyclesCounterStart();
  start_tick = HAL_GetTick();


  /* Perform the inference */
  ret_code = stai_network_run(network_context, STAI_MODE_SYNC);
  if (ret_code != STAI_SUCCESS) {
      ret_code = stai_network_get_error(network_context);
      LC_PRINT("Inference failed with error code %s\r\n", stai_get_return_code_name(ret_code));
  };
  /** End of inference */
  
  /** Continue profiling */
  end_dwt = cyclesCounterEnd();
  total_cycles += end_dwt;
  end_tick = HAL_GetTick();
  dwtCyclesToTime(end_dwt, &t);

  LC_PRINT(" duration DWT    : %d.%03d ms\r\n", t.s * 1000 + t.ms, t.us);
  LC_PRINT(" duration SysTick: %" PRIu32" ms\r\n", end_tick - start_tick);
  LC_PRINT(" CPU cycles      : %" PRIu32 "\r\n", end_dwt);
  LC_PRINT(" CPU cycles (avg): %" PRIu32 "\r\n", total_cycles / ++inference_nb);
  LC_PRINT(" Inference done in %" PRIu32" ms\r\n", end_tick - start_tick);

  return ret_code;
}

/* === Binary Frame Receiver (v3.0 - 692B frames) ================================ */
#define RX_BUF_SIZE   4096
#define FRAME_SIZE    692
#define IQ_PER_FRAME  158

char rx_buf[RX_BUF_SIZE];
volatile uint16_t rx_wp = 0;          /* USART1 IRQ writes here */
volatile uint16_t rx_parse_pos = 0;   /* TIM6 parser reads from here */

/* Statistics (displayed by TIM6 on LCD) */
volatile int stat_bytes = 0;
volatile int stat_frames_l1 = 0;
volatile int stat_frames_l2 = 0;
volatile int stat_pairs = 0;
volatile int stat_crc_err = 0;
volatile int stat_dropped = 0;  /* frames dropped due to busy */

/* New-pair flag: TIM6 parser sets, main loop clears */
volatile int new_pair_ready = 0;

/* Accumulate two lines per IQ pair */
static int16_t iq_pair_buf[316];
static int pending_line1 = 0;

/* Hankel matrix data (40×40 complex) — computed in main loop context */
static float hankel_re[1600];
static float hankel_im[1600];

/* Compute Hankel 40×40 complex matrix from 316 int16 IQ data */
static void compute_hankel(const int16_t *pair)
{
  float diff_re[79], diff_im[79];
  for(int n = 0; n < 79; n++)
  {
    float l_re = (float)pair[2*n];
    float l_im = (float)pair[2*n + 1];
    float r_re = (float)pair[158 + 2*n];
    float r_im = (float)pair[159 + 2*n];
    diff_re[n] = r_re * l_re + r_im * l_im;
    diff_im[n] = r_im * l_re - r_re * l_im;
  }
  for(int m = 0; m < 40; m++)
    for(int k = 0; k < 40; k++)
    {
      hankel_re[m * 40 + k] = diff_re[m + k];
      hankel_im[m * 40 + k] = diff_im[m + k];
    }
}

/* Draw 40×40 phase color map — renders 1600 cells directly to LCD.
   Only called when new IQ data is available (once per ~7s inference cycle).
   No flicker because no other LCD update overlaps this region. */
#define CELL_SZ  10

static void draw_phase_map(void)
{
  for (int m = 0; m < 40; m++)
    for (int k = 0; k < 40; k++)
    {
      float re = hankel_re[m * 40 + k];
      float im = hankel_im[m * 40 + k];
      float ph = atan2f(im, re);
      float t = (ph + 3.14159f) / (2.0f * 3.14159f);
      uint8_t gray = (uint8_t)(t * 255.0f);
      uint32_t color = 0xFF000000 | (gray << 16) | (gray << 8) | gray;
      UTIL_LCD_FillRect(20 + m * CELL_SZ, 60 + k * CELL_SZ,
                        CELL_SZ, CELL_SZ, color);
    }
}

/* Called from TIM6_DAC_IRQHandler every 200ms.
   Only assembles IQ pairs and sets new_pair_ready flag.
   Heavy computation (Hankel, inference) is done in main loop. */
void uart_parse_frames(void)
{
  while(1)
  {
    uint16_t avail;
    __disable_irq();
    if(rx_wp >= rx_parse_pos)
      avail = rx_wp - rx_parse_pos;
    else
      avail = (RX_BUF_SIZE - rx_parse_pos) + rx_wp;
    __enable_irq();

    if(avail < 4) break;

    /* Look for sync 0xAA 0xBB */
    uint8_t b0 = rx_buf[rx_parse_pos];
    uint8_t b1 = rx_buf[(rx_parse_pos + 1) % RX_BUF_SIZE];

    if(b0 != 0xAA || b1 != 0xBB)
    {
      rx_parse_pos = (rx_parse_pos + 1) % RX_BUF_SIZE;
      continue;
    }

    if(avail < FRAME_SIZE) break;

    /* Copy full frame with interrupts disabled to prevent
       USART1 ISR from overwriting ring buffer during copy */
    uint8_t frame[FRAME_SIZE];
    __disable_irq();
    for(int i = 0; i < FRAME_SIZE; i++)
      frame[i] = rx_buf[(rx_parse_pos + i) % RX_BUF_SIZE];
    __enable_irq();

    /* Verify XOR checksum */
    uint8_t ck = 0;
    for(int i = 0; i < FRAME_SIZE - 1; i++) ck ^= frame[i];
    if(ck != frame[FRAME_SIZE - 1])
    {
      stat_crc_err++;
      rx_parse_pos = (rx_parse_pos + 1) % RX_BUF_SIZE;
      continue;
    }

    stat_bytes += FRAME_SIZE;

    /* If main loop is still processing the previous pair, drop this frame.
       This prevents buffer overrun when IQ data arrives faster than AI inference. */
    if(new_pair_ready)
    {
      /* Drop frame — just advance parse position, don't accumulate */
      stat_dropped++;
      rx_parse_pos = (rx_parse_pos + FRAME_SIZE) % RX_BUF_SIZE;
      continue;
    }

    int is_line2 = frame[2];
    int16_t *iq = (int16_t *)&frame[31];

    if(!is_line2)
    {
      stat_frames_l1++;
      memcpy(iq_pair_buf, iq, IQ_PER_FRAME * 2);
      pending_line1 = 1;
    }
    else
    {
      stat_frames_l2++;
      if(pending_line1)
      {
        memcpy(&iq_pair_buf[IQ_PER_FRAME], iq, IQ_PER_FRAME * 2);
        stat_pairs++;
        new_pair_ready = 1;   /* Signal main loop to process */
        pending_line1 = 0;
      }
      else
      {
        memcpy(iq_pair_buf, iq, IQ_PER_FRAME * 2);
        pending_line1 = 1;   /* Treat as line1 for next pair */
      }
    }

    rx_parse_pos = (rx_parse_pos + FRAME_SIZE) % RX_BUF_SIZE;
  }
}

int acquire_and_process_data()
{
  /* USER CODE BEGIN acquire_and_process_data */

  /* Copy IQ pair buffer safely (USART1 ISR / TIM6 parser may write it).
     new_pair_ready stays set during processing to block new frames. */
  int16_t local_iq[316];
  __disable_irq();
  memcpy(local_iq, iq_pair_buf, sizeof(local_iq));
  __enable_irq();

  /* Compute 40×40 complex Hankel matrix in main loop context */
  compute_hankel(local_iq);

  /* Guard: if AI model input buffer is not valid, skip inference */
  if (stai_input[0] == NULL || stai_output[0] == NULL)
  {
    LC_PRINT("AI buffers not ready, skipping inference\r\n");
    HAL_GPIO_TogglePin(GPIOI, LED1_Pin);
    return -1;
  }

  /* Copy Hankel data into AI model input buffer.
     Layout: CHANNEL_FIRST {1, 2, 40, 40}
     Channel 0 = real part, Channel 1 = imaginary part */
  float *inp = (float *)stai_input[0];
  memcpy(inp,           hankel_re, 1600 * sizeof(float));  /* ch0: real */
  memcpy(inp + 1600,    hankel_im, 1600 * sizeof(float));  /* ch1: imag */

  /* Apply Z-score normalization (same as training pipeline) */
  normalize_hankel(inp);

  /* Suspend UART8 + TIM6 during inference to prevent interrupt
     interference with QSPI memory-mapped reads (DTR mode). */
  TIM6->CR1 &= ~TIM_CR1_CEN;
  UART8->CR1 &= ~USART_CR1_RXNEIE;

  /* Run the neural network inference */
  stai_return_code ret = aiRun();

  /* Re-enable UART8 reception and periodic parsing */
  pending_line1 = 0;
  rx_wp = 0;
  rx_parse_pos = 0;
  new_pair_ready = 0;
  UART8->CR1 |= USART_CR1_RXNEIE;
  TIM6->CR1 |= TIM_CR1_CEN;

  if (ret != STAI_SUCCESS) {
    LC_PRINT("aiRun failed: %s\r\n", stai_get_return_code_name(ret));
    return -2;
  }

  HAL_GPIO_TogglePin(GPIOI, LED1_Pin);
  return 0;
  /* USER CODE END acquire_and_process_data */
}

int post_process()
{
  /* USER CODE BEGIN post_process */
  /* Output is a single float32 in stai_output[0] — read by main_loop */
  return 0;
  /* USER CODE END post_process */
}



/* AI initialization status — set by STM32CubeAI_Studio_AI_Init, checked everywhere */
static int ai_ready = 0;

/*
 * Run inference on a single test sample. Returns elapsed ms, or -1 on error.
 * Fills *out_meters with denormalized distance in meters.
 */
static int ai_run_sample(int idx, float *out_meters)
{
  if (!ai_ready || idx >= NUM_TEST_SAMPLES) return -1;

  float *inp = (float *)stai_input[0];
  memcpy(inp, test_samples[idx], TEST_HANKEL_SIZE * sizeof(float));
  normalize_hankel(inp);

  uint32_t t0 = HAL_GetTick();
  stai_return_code ret = aiRun();
  int elapsed = (int)(HAL_GetTick() - t0);

  if (ret == STAI_SUCCESS) {
    float raw = *(float *)stai_output[0];
    *out_meters = denormalize_output(raw);
    LC_PRINT("Sample %d: %dms  raw=%.4f  dist=%.4f m  (label=%.4f)\r\n",
             idx, elapsed, (double)raw, (double)*out_meters,
             (double)test_labels[idx]);
    return elapsed;
  }
  LC_PRINT("Sample %d FAILED: %s\r\n", idx, stai_get_return_code_name(ret));
  return -1;
}


/* Private variables for LCD display */
static uint32_t LCD_X_Size = 0;
static uint32_t LCD_Y_Size = 0;


/*
 * Main loop: run self-test first, then wait for IQ data pairs.
 */
void main_loop() {
  /* USER CODE BEGIN main_loop */
  BSP_LCD_GetXSize(0, &LCD_X_Size);
  BSP_LCD_GetYSize(0, &LCD_Y_Size);

  UTIL_LCD_Clear(0xFF223344);  /* soft dark blue background */
  UTIL_LCD_SetBackColor(0xFF223344);
  UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);
  UTIL_LCD_SetFont(&Font16);
  UTIL_LCD_SetTextColor(0xFFDDDDDD);
  UTIL_LCD_DisplayStringAt(430, LINE(2), (uint8_t *)"AI Self-Test Mode", LEFT_MODE);
  if(ai_ready) {
    UTIL_LCD_SetTextColor(0xFF77AA77);
    UTIL_LCD_DisplayStringAt(430, LINE(3), (uint8_t *)"AI: DTR+Cached", LEFT_MODE);
  } else {
    UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_RED);
    UTIL_LCD_DisplayStringAt(430, LINE(3), (uint8_t *)"AI: FAILED", LEFT_MODE);
  }
  UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);

  /* === Phase 1: Multi-sample self-test (cold vs warm cache) === */
  if (ai_ready) {
    UTIL_LCD_DisplayStringAt(430, LINE(5), (uint8_t *)"Multi-test: 3 samples", LEFT_MODE);
    HAL_Delay(500);

    int times[3] = {0};
    float dists[3] = {0};
    int pass = 1;
    char msg[60];

    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
      snprintf(msg, sizeof(msg), "Sample %d/3...", i + 1);
      UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_YELLOW);
      UTIL_LCD_DisplayStringAt(430, LINE(6), (uint8_t *)msg, LEFT_MODE);

      int t = ai_run_sample(i, &dists[i]);
      if (t < 0) { pass = 0; times[i] = -1; }
      else times[i] = t;

      /* Show result for this sample */
      snprintf(msg, sizeof(msg), "S%d: %dms %.3fm (L:%.2f)", i+1, times[i],
               (double)dists[i], (double)test_labels[i]);
      UTIL_LCD_SetTextColor(t >= 0 ? UTIL_LCD_COLOR_GREEN : UTIL_LCD_COLOR_RED);
      UTIL_LCD_DisplayStringAt(430, LINE(7 + i), (uint8_t *)msg, LEFT_MODE);
    }

    /* Summary */
    UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);
    snprintf(msg, sizeof(msg), "Times: %d %d %d ms  %s",
             times[0], times[1], times[2],
             pass ? "PASS" : "FAIL");
    UTIL_LCD_DisplayStringAt(430, LINE(11), (uint8_t *)msg, LEFT_MODE);
    if (pass) HAL_GPIO_TogglePin(GPIOI, LED1_Pin);
  } else {
    UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_RED);
    UTIL_LCD_DisplayStringAt(430, LINE(5), (uint8_t *)"AI not ready — skip test", LEFT_MODE);
  }

  /* Clear Phase 1 text from right panel */
  UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_BLUE);
  UTIL_LCD_FillRect(420, 0, 380, 480, 0xFF223344);

  /* Right panel labels — all y positions use Font16-based pixel offsets */
  #define RY(y)  (y * 16)

  UTIL_LCD_SetFont(&Font20);
  UTIL_LCD_SetTextColor(0xFFCC8800);  /* warm orange */
  UTIL_LCD_DisplayStringAt(428, 50, (uint8_t *)"Distance", LEFT_MODE);
  UTIL_LCD_SetFont(&Font16);
  UTIL_LCD_SetTextColor(0xFF888888);  /* soft gray */
  UTIL_LCD_DisplayStringAt(428, 100, (uint8_t *)"--", LEFT_MODE);
  UTIL_LCD_SetTextColor(0xFF888888);
  UTIL_LCD_DisplayStringAt(428, 150, (uint8_t *)"Pair:   --", LEFT_MODE);
  UTIL_LCD_DisplayStringAt(428, 180, (uint8_t *)"Time:   --", LEFT_MODE);
  UTIL_LCD_DisplayStringAt(428, 210, (uint8_t *)"Status: --", LEFT_MODE);

  /* === Phase 2: Normal UART processing mode === */
  int last_pairs = 0;
  while (1) {
    if(new_pair_ready && stat_pairs > last_pairs)
    {
      last_pairs = stat_pairs;

      uint32_t t0 = HAL_GetTick();
      int ret = acquire_and_process_data();
      uint32_t elapsed = HAL_GetTick() - t0;

      /* Draw phase map */
      TIM6->CR1 &= ~TIM_CR1_CEN;
      draw_phase_map();
      TIM6->CR1 |= TIM_CR1_CEN;

      char msg[60];

      /* Distance value */
      UTIL_LCD_SetFont(&Font20);
      if(ret == 0)
      {
        float dist_m = denormalize_output(*(float *)stai_output[0]);
        snprintf(msg, sizeof(msg), "%.3f m", (double)dist_m);
        UTIL_LCD_SetTextColor(0xFF66BB66);  /* soft green */
      }
      else
      {
        snprintf(msg, sizeof(msg), "FAIL");
        UTIL_LCD_SetTextColor(0xFFCC6666);  /* soft red */
      }
      UTIL_LCD_DisplayStringAt(428, 100, (uint8_t *)msg, LEFT_MODE);
      UTIL_LCD_SetFont(&Font16);

      /* Pair */
      snprintf(msg, sizeof(msg), "Pair:   %d", stat_pairs);
      UTIL_LCD_SetTextColor(0xFF66AACC);  /* soft cyan */
      UTIL_LCD_DisplayStringAt(428, 150, (uint8_t *)msg, LEFT_MODE);

      /* Time */
      snprintf(msg, sizeof(msg), "Time:   %lums", elapsed);
      UTIL_LCD_SetTextColor(0xFFCCAA44);  /* soft yellow */
      UTIL_LCD_DisplayStringAt(428, 180, (uint8_t *)msg, LEFT_MODE);

      /* Status */
      UTIL_LCD_SetTextColor(ret == 0 ? 0xFF66BB66 : 0xFFCC6666);
      UTIL_LCD_DisplayStringAt(428, 210, (uint8_t *)(ret == 0 ? "Status: OK" : "Status: FAIL"), LEFT_MODE);

      __disable_irq();
      new_pair_ready = 0;
      __enable_irq();
    }
    else
    {
      HAL_Delay(10);
    }
  }
  /* USER CODE END main_loop */
}


/* ── UART8 initialization (IQ data from sensor via Arduino D0/D1) ───── */
void MX_UART8_Init(void)
{
    huart8.Instance = UART8;
    huart8.Init.BaudRate = 115200;
    huart8.Init.WordLength = UART_WORDLENGTH_8B;
    huart8.Init.StopBits = UART_STOPBITS_1;
    huart8.Init.Parity = UART_PARITY_NONE;
    huart8.Init.Mode = UART_MODE_TX_RX;
    huart8.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart8.Init.OverSampling = UART_OVERSAMPLING_16;
    huart8.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    huart8.Init.ClockPrescaler = UART_PRESCALER_DIV1;
    huart8.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    if(HAL_UART_Init(&huart8) != HAL_OK)
    {
        Error_Handler();
    }
}


/* Entry points --------------------------------------------------------------*/


void STM32CubeAI_Studio_AI_Init(void)
{
    MX_UARTx_Init();   /* USART1 (STLink) — debug LC_PRINT */
    MX_UART8_Init();   /* UART8 (Arduino D0/D1) — IQ data */
    BSP_QSPI_Init_t qspiInit;
                    qspiInit.InterfaceMode=MT25TL01G_QPI_MODE;
                    qspiInit.TransferRate= MT25TL01G_DTR_TRANSFER ;
                    qspiInit.DualFlashMode= MT25TL01G_DUALFLASH_ENABLE;
                    BSP_QSPI_Init(0,&qspiInit);
                    BSP_QSPI_EnableMemoryMappedMode(0);
    BSP_SDRAM_Init(0);

    if(aiInit() == 0) {
      ai_ready = 1;
    } else {
      ai_ready = 0;
      LC_PRINT("*** WARNING: AI init failed ***\r\n");
    }

/* USER CODE BEGIN init */
    /* Enable UART8 RXNE interrupt for IQ data reception */
    /* (USART1 kept for LC_PRINT debug via HAL_UART_Transmit, ISR uses HAL) */
    UART8->CR1 |= USART_CR1_RXNEIE;

    /* Configure TIM6 for periodic LCD refresh (every 200ms) */
    __HAL_RCC_TIM6_CLK_ENABLE();
    TIM6->PSC = 10000 - 1;   /* 100MHz / 10000 = 10kHz */
    TIM6->ARR = 2000 - 1;    /* 10kHz / 2000 = 5Hz = 200ms */
    TIM6->DIER |= TIM_DIER_UIE; /* Enable update interrupt */
    HAL_NVIC_SetPriority(TIM6_DAC_IRQn, 6, 0);
    HAL_NVIC_EnableIRQ(TIM6_DAC_IRQn);
    TIM6->CR1 |= TIM_CR1_CEN;   /* Start timer */

    /* Initialize LCD DSI display */
    if(BSP_LCD_Init(0, LCD_ORIENTATION_LANDSCAPE) != BSP_ERROR_NONE)
    {
      Error_Handler();
    }
    UTIL_LCD_SetFuncDriver(&LCD_Driver);
    UTIL_LCD_SetLayer(0);
    /* USER CODE END init */
}

void STM32CubeAI_Studio_AI_Process(void)
{
    main_loop();
} 

void STM32CubeAI_Studio_AI_Deinit(void)
{
    aiDeinit();
} 


#ifdef __cplusplus
}
#endif
