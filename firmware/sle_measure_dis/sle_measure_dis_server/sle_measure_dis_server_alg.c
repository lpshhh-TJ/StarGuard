/**
 * Copyright (c) HiSilicon (Shanghai) Technologies Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: sle adv config for sle uart server. \n
 *
 * History: \n
 * 2023-07-17, Create file. \n
 * Has been changed by lpshhh-TJ \n
 */
#include "sle_measure_dis_server.h"
#include "sle_common.h"
#include "slem_smooth.h"
#include "slem_alg_smooth_dis.h"
#include "tcxo.h"
#include "sle_measure_dis_server_alg.h"

#define TOF_DEFAULT 2070
#define DIS_ALG_MODE 4
#define DIS_ALG_RSSI_LIMIT (-120)
#define DIS_ALG_THRESHOULD_COND2 20
#define DIS_ALG_R_START 2

uint8_t g_iq_debug_flag = 0;
extern uint8_t g_measure_dis_server_addr[];
uint32_t counter = 0;

// Replaced single variable with Ring Buffer
#define LOCAL_IQ_BUFFER_SIZE 8
measure_dis_stored_iq_data_t g_md_local_iq_buffer[LOCAL_IQ_BUFFER_SIZE];
uint8_t g_md_local_iq_buffer_idx = 0; // Index for next write

// Kept for partial assembly (multi-packet assembly for a single IQ report)
measure_dis_stored_iq_data_t g_md_local_iq_assembly; 
uint8_t g_next_idx;


void slem_posalg_set_base_para(slem_alg_para_dis *alg_para, uint8_t key_id)
{
    unused(key_id);
    alg_para->calib_val = 0;
    alg_para->tof_calib = TOF_DEFAULT;
    alg_para->ranging_method = DIS_ALG_MODE;
}

void measure_dis_print_cal_dis(float dist_first, float dist_ori, float prob, uint32_t time)
{
    unused(dist_ori);
    unused(prob);
    osal_printk(
        "SLEM get distance done. distance:%d.%03d, time = %d ms. \r\n",
        (int32_t)(dist_first), ((uint32_t)(dist_first * MEASURE_DIS_NUM_CARRY_1000) % MEASURE_DIS_NUM_CARRY_1000),
        (uint32_t)(time));
}

static void measure_dis_posalg_set_distance(slem_smoothed_dis_result *dis,
    measure_dis_profile_msg_dis_t *measure_dis_temp)
{
    measure_dis_temp->dist_first = dis->dis_smoothed;
    measure_dis_temp->dist_second = dis->dis_ori;
    measure_dis_temp->dist_double = dis->dis_slight_smoothed;
    measure_dis_temp->high = dis->height;
    measure_dis_temp->prob = dis->prob;
    measure_dis_temp->rssi = dis->rssi;
    measure_dis_temp->smooth_num = dis->smooth_num;
}

void measure_dis_posalg_set_base_para(slem_alg_para_dis *alg_para)
{
    alg_para->calib_val = 0;
    alg_para->tof_calib = TOF_DEFAULT;
    alg_para->ranging_method = DIS_ALG_MODE;
}

void measure_dis_posalg_get_distance(measure_dis_profile_msg_dis_t *measure_dis_temp,
                                     sle_channel_sounding_iq_trans_t *remote_iq_data,
                                     measure_dis_stored_iq_data_t *local_iq_data)
{
    slem_alg_para_dis *alg_para = (slem_alg_para_dis *)osal_kmalloc(sizeof(slem_alg_para_dis), 0);

    slem_posalg_set_base_para(alg_para, 0);
    alg_para->rssi_rtd = local_iq_data->rssi;
    alg_para->rssi_dut = remote_iq_data[0].rssi;
#if (defined(GLE_CS_MODE3_SUPPORT))
    alg_para->tof_rtd = local_iq_data->tof_result;
    alg_para->tof_dut = remote_iq_data[0].tof_result;
#endif
    alg_para->iq_rtd = (slem_alg_iq *)&(local_iq_data->data[0]);
    alg_para->iq_dut = (slem_alg_iq *)&(remote_iq_data[0].data[0]);
    
    // g_md_store_local_iq_complete logic removed as it's handled by buffer

    slem_para_pair para_limit = {.rssi_limit = DIS_ALG_RSSI_LIMIT,
                                 .threshold_cond2 = DIS_ALG_THRESHOULD_COND2,
                                 .r_start = DIS_ALG_R_START};
    alg_para->para_limit = para_limit;
    alg_para->key_id = 0;
    /* remote iq 是 dut， local iq 是 rtd */
    uint32_t time_start = (uint32_t)uapi_tcxo_get_ms();

    alg_para->cur_count = counter;
    alg_para->cur_time = time_start;
    slem_smoothed_dis_result result_dist;
    slem_alg_calc_smoothed_dis(&result_dist, alg_para);
    counter++;
    uint32_t time_end = (uint32_t)uapi_tcxo_get_ms();
    measure_dis_print_cal_dis(result_dist.dis_smoothed, result_dist.dis_ori,
                              result_dist.prob, time_end - time_start);

    measure_dis_posalg_set_distance(&result_dist, measure_dis_temp);
    osal_kfree(alg_para);
}

void measure_dis_print_iq_data(uint8_t num, uint8_t role, sle_channel_sounding_iq_trans_t *report)
{
    // Force print for debugging
    // if (!g_iq_debug_flag) {
    //     return;
    // }
    osal_printk("[iq_data]%d,%d,%d,%d,%d,%d,0x%x,",
                g_measure_dis_server_addr[0], num, role, (uint32_t)uapi_tcxo_get_ms(), report->timestamp_sn, report->samp_cnt, report->rssi);
    sle_channel_sounding_qte_trans_t *data = report->data;
    uint16_t i = 0;
    // Optimize: print 4 samples per line to reduce syscall overhead
    while (i < POSALG_DATA_NUM) {
        if (i + 3 < POSALG_DATA_NUM) {
             osal_printk("[%d]:0x%x,0x%x;[%d]:0x%x,0x%x;[%d]:0x%x,0x%x;[%d]:0x%x,0x%x;", 
                i, data[i].i_data, data[i].q_data,
                i+1, data[i+1].i_data, data[i+1].q_data,
                i+2, data[i+2].i_data, data[i+2].q_data,
                i+3, data[i+3].i_data, data[i+3].q_data);
             i += 4;
        } else {
             osal_printk("[%d]:0x%x,0x%x", i, data[i].i_data, data[i].q_data);
             i++;
             if (i != POSALG_DATA_NUM) {
                 osal_printk(";");
             }
        }
    }
    osal_printk(".\r\n");
}

errcode_t measure_dis_store_local_iq(sle_channel_sounding_iq_report_t *report)
{
    if (report->report_idx == 0) {
        g_md_local_iq_assembly.samp_cnt = 0;
        g_md_local_iq_assembly.rssi = report->rssi[0];
        g_md_local_iq_assembly.es_sn = report->es_sn;
        g_md_local_iq_assembly.timestamp_sn = report->timestamp_sn;
#if (defined(GLE_CS_MODE3_SUPPORT))
        g_md_local_iq_assembly.tof_result = report->tof_result;
#endif
        g_next_idx = 0;
    }
    if ((report->report_idx >= MEASURE_DIS_IQ_REPORT_CNT_MAX ||
         report->timestamp_sn != g_md_local_iq_assembly.timestamp_sn) ||
        (g_next_idx != report->report_idx)) {
        osal_printk("MEASURE_DIS STORE LOCAL IQ FAIL. report_idx:%d, timestamp:%d, g_next_idx:%d. \r\n",
                    report->report_idx, report->timestamp_sn, g_next_idx);
        return ERRCODE_INVALID_PARAM;
    }
    g_md_local_iq_assembly.samp_cnt += report->samp_cnt;
    uint8_t offset = report->report_idx * SLE_CS_IQ_REPORT_COUNT;
    for (uint8_t i = 0; i < report->samp_cnt; i++) {
        g_md_local_iq_assembly.data[offset + i].i_data = report->i_data[i];
        g_md_local_iq_assembly.data[offset + i].q_data = report->q_data[i];
    }
    g_next_idx = report->report_idx + 1;

#if (defined(GLE_CS_MODE3_SUPPORT))
    osal_printk("local tof_result = %d.%03d \r\n", report->tof_result / MEASURE_DIS_NUM_CARRY_1000,
                report->tof_result % MEASURE_DIS_NUM_CARRY_1000);
#endif

    return ERRCODE_SUCC;
}

errcode_t measure_dis_proc_local_iq(uint16_t conn_id, sle_channel_sounding_iq_report_t *report)
{
    unused(conn_id);
    if (report->report_idx == 0) {
        osal_printk("RECEIVE LOCAL IQ. timestamp_sn:%d \r\n",  report->timestamp_sn);
    }
    errcode_t ret = ERRCODE_SUCC;

    ret = measure_dis_store_local_iq(report);
    if (ret != ERRCODE_SUCC) {
        return ERRCODE_INVALID_PARAM;
    }

    if (report->report_idx + 1 == MEASURE_DIS_IQ_REPORT_CNT_MAX) {
        // Assembly complete, move to Ring Buffer
        uint8_t idx = g_md_local_iq_buffer_idx;
        if (memcpy_s(&g_md_local_iq_buffer[idx], sizeof(measure_dis_stored_iq_data_t), 
                     &g_md_local_iq_assembly, sizeof(measure_dis_stored_iq_data_t)) != EOK) {
             osal_printk("Memcpy check fail\n");
        }
        
        osal_printk("Local IQ stored in Buffer[%d], ts=%d\r\n", idx, g_md_local_iq_buffer[idx].timestamp_sn);
        
        // Advance Ring Buffer Index
        g_md_local_iq_buffer_idx = (g_md_local_iq_buffer_idx + 1) % LOCAL_IQ_BUFFER_SIZE;
    }
    return ret;
}

errcode_t measure_dis_proc_remote_iq(uint16_t conn_id, sle_channel_sounding_iq_trans_t *report)
{
    unused(conn_id);
    // measure_dis_print_iq_data(0, 1, report);
    
    // Find matching Local IQ in Buffer
    int matched_idx = -1;
    for (int i = 0; i < LOCAL_IQ_BUFFER_SIZE; i++) {
        // Check timestamp match with tolerance (+- 20)
        uint32_t diff = (g_md_local_iq_buffer[i].timestamp_sn > report->timestamp_sn) ? 
                        (g_md_local_iq_buffer[i].timestamp_sn - report->timestamp_sn) : 
                        (report->timestamp_sn - g_md_local_iq_buffer[i].timestamp_sn);
        
        if (diff <= 20) {
            matched_idx = i;
            break;
        }
    }
    
    if (matched_idx == -1) {
        osal_printk("No match found for remote ts:%d. Discarding.\r\n", report->timestamp_sn);
        return ERRCODE_INVALID_PARAM;
    }

    osal_printk("Matched! Buffer[%d] local ts = %d, remote ts = %d.\r\n",
                matched_idx, g_md_local_iq_buffer[matched_idx].timestamp_sn, report->timestamp_sn);

    // Print Local IQ when matching complete
    measure_dis_print_iq_data(0, 0, (sle_channel_sounding_iq_trans_t *)(&g_md_local_iq_buffer[matched_idx]));
    
    // Print Remote IQ
    measure_dis_print_iq_data(0, 1, report);
    
    // Calculate if needed
    // measure_dis_profile_msg_dis_t measure_dis_dis_temp = { report->timestamp_sn, 0, 0, 0, 0, 0, 0, 0};
    // measure_dis_posalg_get_distance(&measure_dis_dis_temp, (sle_channel_sounding_iq_trans_t *)report, &g_md_local_iq_buffer[matched_idx]);

    return ERRCODE_SUCC;
}

void measure_dis_read_local_cs_caps_cb(sle_channel_sounding_caps_t *caps, errcode_t status)
{
    unused(caps);
    osal_printk("SLEM READ LOCAL CAPS. status:%d. \r\n", status);
}

void measure_dis_read_remote_cs_caps_cb(uint16_t conn_id, sle_channel_sounding_caps_t *caps, errcode_t status)
{
    unused(caps);
    unused(conn_id);
    osal_printk("SLEM READ REMOTE CAPS. status:%d. \r\n", status);
}

void measure_dis_set_cs_param_cb(uint16_t conn_id, errcode_t status)
{
    unused(conn_id);
    osal_printk("SLEM SET PARAM. status:%d. \r\n", status);
}

void measure_dis_cs_state_changed_cb(uint8_t slem_status, errcode_t status)
{
    unused(slem_status);
    osal_printk("SLEM STATE CHANGED. status:%d. \r\n", status);
}

void measure_dis_cs_iq_report_cb(uint16_t conn_id, sle_channel_sounding_iq_report_t *report)
{
    errcode_t ret = ERRCODE_FAIL;

    sle_channel_sounding_iq_report_t *report_local = NULL;
    uint32_t len = sizeof(sle_channel_sounding_iq_report_t);
    report_local = (sle_channel_sounding_iq_report_t *)osal_kmalloc(len, 0);
    if (report_local == NULL) {
        return;
    }
    if (memcpy_s(report_local, sizeof(sle_channel_sounding_iq_report_t), report, len) != EOK) {
        osal_kfree(report_local);
        return;
    }
    measure_dis_msg_node_t msg_node = {conn_id, SLEM_MSG_LOCAL_IQ, 0, report_local};
    ret = sle_measure_dis_msg_add(&msg_node);
    if (ret != ERRCODE_SUCC) {
        osal_kfree(report_local);
    }
}

errcode_t measure_dis_reg_callbacks(void)
{
    sle_hadm_callbacks_t scd_cbks = {0};

    scd_cbks.read_local_cs_caps_cb = measure_dis_read_local_cs_caps_cb;
    scd_cbks.read_remote_cs_caps_cb = measure_dis_read_remote_cs_caps_cb;
    scd_cbks.cs_state_changed_cb = measure_dis_cs_state_changed_cb;
    scd_cbks.cs_iq_report_cb = measure_dis_cs_iq_report_cb;

    errcode_t ret = sle_hadm_register_callbacks(&scd_cbks);
    osal_printk("SCD SET CALLBACK DONE. ret = %x.\r\n", ret);
    return ret;
}

errcode_t measure_dis_recv_local_iq(measure_dis_msg_node_t *msg_node)
{
    /* key send to anchor */
    errcode_t ret = measure_dis_proc_local_iq(msg_node->conn_id, (sle_channel_sounding_iq_report_t *)(msg_node->data));
    osal_kfree(msg_node->data);
    osal_event_write(&measure_dis_evt, MEASURE_DIS_MSG_EVENT);
    return ret;
}

errcode_t measure_dis_recv_remote_iq(measure_dis_msg_node_t *msg_node)
{
    osal_printk("slem recv remote iq.\n");
    errcode_t ret = measure_dis_proc_remote_iq(msg_node->conn_id, (sle_channel_sounding_iq_trans_t *)(msg_node->data));
    osal_kfree(msg_node->data);
    osal_event_write(&measure_dis_evt, MEASURE_DIS_MSG_EVENT);
    return ret;
}

errcode_t measure_dis_match_msg(measure_dis_msg_node_t *msg_node)
{
    errcode_t ret = ERRCODE_FAIL;
    osal_printk("---enter sle_slem_msg_proc, type:%d---\n", msg_node->type);
    switch (msg_node->type) {
        case SLEM_MSG_LOCAL_IQ:
            ret = measure_dis_recv_local_iq(msg_node);
            break;
        case SLEM_MSG_REMOTE_IQ:
            ret = measure_dis_recv_remote_iq(msg_node);
            break;
        default:
            break;
    }
    return ret;
}

errcode_t measure_dis_remote_iq(uint16_t len, uint8_t *value)
{
    UNUSED(len);
    errcode_t ret = ERRCODE_FAIL;
    sle_channel_sounding_iq_trans_t *report = NULL;
    report = (sle_channel_sounding_iq_trans_t *)osal_kmalloc(sizeof(sle_channel_sounding_iq_trans_t), 0);
    if (report == NULL) {
        return ERRCODE_MALLOC;
    }
    if (memcpy_s(report, sizeof(sle_channel_sounding_iq_trans_t), value, len) != EOK) {
        osal_kfree(report);
        return ERRCODE_MEMCPY;
    }
    osal_printk("RECEIVE REMOTE IQ. timestamp_sn:%d \r\n", report->timestamp_sn);

    measure_dis_msg_node_t msg_node = {g_measure_dis_conn_id, SLEM_MSG_REMOTE_IQ, 0, report};
    ret = sle_measure_dis_msg_add(&msg_node);
    if (ret != ERRCODE_SUCC) {
        osal_kfree(report);
    }

    return ret;
}