/**
 * Copyright (c) HiSilicon (Shanghai) Technologies Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: SLE MEASURE_DIS sample of client. \n
 *
 * History: \n
 * 2023-04-03, Create file. \n
 * Has been changed by lpshhh-TJ \n
 */
#ifndef SLE_MEASURE_DIS_CLIENT_H
#define SLE_MEASURE_DIS_CLIENT_H

#include <stdint.h>
#include <securec.h>
#include "errcode.h"
#include "std_def.h"
#include "soc_osal.h"
#include "common_def.h"
#include "sle_ssap_client.h"
#include "sle_common.h"
#include <stdbool.h>

#define MEASURE_DIS_NUM_CARRY_10_7     10000000
#define MEASURE_DIS_NUM_CARRY_1000     1000
#define MEASURE_DIS_NUM_CARRY_100      100
#define MEASURE_DIS_NUM_CARRY_10       10
#define MEASURE_DIS_NUM_CARRY_2        2

#define SLEM_CONNET_INVAILD 0xFF
#define MAX_SERVER_NUM 4 /* 最大连接 Server 数量 */

typedef enum {
    STATE_IDLE,                 // 空闲
    STATE_SELECT_TARGET,        // 选择目标
    STATE_START_RANGING,        // 开启测距
    STATE_WAIT_LOCAL_IQ,        // 等待IQ上报
    STATE_SEND_REMOTE_IQ,       // 发送IQ数据
    STATE_ERROR_RECOVERY        // 错误恢复
} client_rang_state_t;

typedef struct {
    uint16_t conn_ids[MAX_SERVER_NUM];     // 连接ID列表
    sle_addr_t server_addrs[MAX_SERVER_NUM]; // 对端地址列表
    bool is_connected[MAX_SERVER_NUM];     // 连接状态列表
    uint8_t conn_num;                      // 当前连接总数
    
    uint8_t current_idx;                   // 当前轮询的索引
    client_rang_state_t current_state;     // 当前状态
    uint32_t state_start_timestamp;        // 状态开始时间戳
    
    bool iq_received_flag;                 // IQ数据接收标志
} client_context_t;

extern client_context_t g_client_ctx;
void measure_dis_client_task_process(void);
void measure_dis_client_set_iq_received(void);

typedef enum {
    SLEM_PROFILE_MSG_IQ = 0xFFFFFFEA,
} slem_profile_msg_type_t;

typedef struct {
    uint32_t        type;                             /*!< 消息类型 */
    uint32_t        len;                              /*!< 消息类型 */
    uint8_t         data[0];
} measure_ids_msg_t;

typedef struct server_data {
    uint8_t property_uuid[SLE_UUID_LEN];
    uint8_t server_uuid[SLE_UUID_LEN];
    uint8_t service_uuid[SLE_UUID_LEN];
    uint16_t begin_hdl;
    uint16_t end_hdl;
    uint16_t property_handle;
} measure_dis_server_data_t;

#define check_rc_return_rc(rc, err)                                                             \
    do {                                                                                         \
        if ((rc) != ERRCODE_SUCC) {                                                              \
            osal_printk("CARKEY ERROR: %s fail!: call %s return 0x%x!\n", err, __FUNCTION__, rc);   \
        }                                                                                        \
    } while (0)

int measure_dis_client_write_server(uint32_t type, uint8_t *data, uint32_t data_len, uint16_t conn_id);
int measure_dis_client_init(void);
int measure_dis_start_scan(void);

#endif