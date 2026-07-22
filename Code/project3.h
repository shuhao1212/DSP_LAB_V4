#ifndef _PROJECT3_H_
#define _PROJECT3_H_

#include "driver_include.h"
#include "user_include.h"
#include "weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ============================================================
// 固定布局宏定义 - 防偏移专用
// ============================================================
#define P3_LCD_W                800
#define P3_LCD_H                480
#define P3_TEXT_AREA_X          100
#define P3_TEXT_AREA_Y          180
#define P3_TEXT_AREA_W          600
#define P3_TEXT_AREA_H          120
#define P3_TEXT_X               200  // 文字绘制起点X（绝对坐标）
#define P3_TEXT_Y               215  // 文字绘制起点Y（绝对坐标）

// 坐标裁剪宏
#define P3_CLAMP_X(x)           (((x) < 0) ? 0 : (((x) >= P3_LCD_W) ? (P3_LCD_W - 1) : (x)))
#define P3_CLAMP_Y(y)           (((y) < 0) ? 0 : (((y) >= P3_LCD_H) ? (P3_LCD_H - 1) : (y)))

// LCD刷新控制
#define P3_REFRESH_INTERVAL     25  // 每25帧刷新一次（约500ms）

// ============================================================
// UI 美化布局宏定义
// ============================================================
// 顶部标题栏
#define P3_HEADER_Y              0
#define P3_HEADER_H              56
#define P3_HEADER_BG             ClrDarkSlateGray
#define P3_HEADER_SEPARATOR_Y    (P3_HEADER_Y + P3_HEADER_H - 1)
#define P3_TITLE_X               20
#define P3_TITLE_Y               10
#define P3_TITLE_FONT            g_sFontCm28

// 状态指示灯
#define P3_STATUS_DOT_X          (P3_LCD_W - 40)
#define P3_STATUS_DOT_Y          28
#define P3_STATUS_DOT_R          8
#define P3_STATUS_DOT_RING_R     12

// 中央卡片区域
#define P3_CARD_X                60
#define P3_CARD_Y                90
#define P3_CARD_W                680
#define P3_CARD_H                240
#define P3_CARD_MARGIN           24

// 卡片内文字位置
#define P3_RESULT_TEXT_Y         (P3_CARD_Y + 45)
#define P3_CONFIDENCE_TEXT_Y     (P3_CARD_Y + 140)
#define P3_CONFIDENCE_BAR_Y      (P3_CARD_Y + 175)
#define P3_CONFIDENCE_BAR_W      400
#define P3_CONFIDENCE_BAR_H      12
#define P3_CONFIDENCE_BAR_X      ((P3_LCD_W - P3_CONFIDENCE_BAR_W) / 2)

// 底部状态栏
#define P3_BOTTOM_BAR_Y          440
#define P3_BOTTOM_BAR_H          40
#define P3_BOTTOM_BAR_BG         ClrBlack
#define P3_BOTTOM_SEPARATOR_Y    P3_BOTTOM_BAR_Y

#define PROJECT3_PI                         3.14159265358979323846f

/* Hardware stream: use 20 kHz ADC, then decimate 5 -> 4 to match the PC training rate of 16 kHz. */
#define PROJECT3_ADC_RATE                   ADC_20KHZ
#define PROJECT3_DAC_RATE                   DAC_20KHZ
#define PROJECT3_BLOCK_SAMPLES              ADC_SAMPLE_1024
#define PROJECT3_DAC_CHANNEL_MASK           DAC_CHANNEL_1
#define PROJECT3_HW_SAMPLE_RATE             20000
#define PROJECT3_MODEL_SAMPLE_RATE          16000
#define PROJECT3_INPUT_GAIN                 2.0f   /* ADC增益补偿：DSP信号偏弱，放大2倍匹配PC训练电平 */

/* One command is normalized to the same 1-second waveform used by train_bcresnet.py. */
#define PROJECT3_RAW_MAX_SAMPLES            PROJECT3_HW_SAMPLE_RATE
#define PROJECT3_MODEL_SAMPLES              16000  /* 1秒 @16kHz */
#define PROJECT3_VAD_FRAME_LEN              400
#define PROJECT3_VAD_HOP                    200
#define PROJECT3_WIN_SIZE                   480
#define PROJECT3_HOP_SIZE                   160
#define PROJECT3_FFT_LEN                    512
#define PROJECT3_FREQ_NUM                   (PROJECT3_FFT_LEN / 2 + 1)
#define PROJECT3_MELS_NUM                   40
#define PROJECT3_MODEL_FRAMES               101    /* 标准 1 秒 */
#define PROJECT3_CMD_COUNT                  13

#define PROJECT3_UI_TEXT_LEN                64
#define PROJECT3_RESULT_TEXT_LEN            32
#define PROJECT3_PASS_THROUGH_ENABLE        0
#define PROJECT3_INFERENCE_CONF_THRESHOLD   0.35f
#define PROJECT3_CONFIDENCE_DISPLAY_THRESHOLD  0.50f
#define PROJECT3_DOWN_CONF_THRESHOLD        0.35f
#define PROJECT3_DOWN_MAX_SAMPLES           PROJECT3_HW_SAMPLE_RATE
#define PROJECT3_MIN_UTTERANCE_SAMPLES      (PROJECT3_HW_SAMPLE_RATE / 5)
#define PROJECT3_POST_INFER_IGNORE_BLOCKS   8
#define PROJECT3_RESULT_HOLD_BLOCKS         20
#define PROJECT3_REJECT_HOLD_BLOCKS         50

#define PROJECT3_CENTER_X                   400
#define PROJECT3_CENTER_Y                   215
#define PROJECT3_CENTER_W                   520
#define PROJECT3_CENTER_H                   150

#define PROJECT3_BN_EPS                     1.0e-5f
#define PROJECT3_ACT_MAX                    (16 * 20 * PROJECT3_MODEL_FRAMES)
#define PROJECT3_BN_MERGED_CHANNELS_MAX     32

/* ============================================================
 *  预合并 BatchNorm 参数 (推理加速)
 *  merged_w[c] = gamma[c] / sqrt(var[c] + eps)
 *  merged_b[c] = beta[c] - mean[c] * merged_w[c]
 *  推理时 BN 简化为: y = x * merged_w[c] + merged_b[c]
 *  消除每次推理的 sqrt/div 运算, 约 20% 加速
 * ============================================================ */
#pragma DATA_ALIGN(g_project3_merged_bn, 8)
static float g_project3_merged_bn_w[17][PROJECT3_BN_MERGED_CHANNELS_MAX];
static float g_project3_merged_bn_b[17][PROJECT3_BN_MERGED_CHANNELS_MAX];
static unsigned char g_project3_merged_bn_ready = 0;

/* BN 层索引 */
enum {
    P3_BN_IDX_conv1 = 0,
    P3_BN_IDX_l1b1, P3_BN_IDX_l1b2, P3_BN_IDX_l1b3, P3_BN_IDX_l1sc,
    P3_BN_IDX_l2b1, P3_BN_IDX_l2b2, P3_BN_IDX_l2b3, P3_BN_IDX_l2sc,
    P3_BN_IDX_l3b1, P3_BN_IDX_l3b2, P3_BN_IDX_l3b3, P3_BN_IDX_l3sc,
    P3_BN_IDX_dw, P3_BN_IDX_pw, P3_BN_IDX_conv2, P3_BN_IDX_expand
};

typedef enum {
    PROJECT3_APP_BOOT = 0,
    PROJECT3_APP_LISTENING,
    PROJECT3_APP_SPEECH,
    PROJECT3_APP_INFERENCING,
    PROJECT3_APP_RESULT,
    PROJECT3_APP_ERROR
} PROJECT3_APP_STATE;

typedef enum {
    PROJECT3_MODEL_READY = 0
} PROJECT3_MODEL_STATE;

typedef enum {
    PROJECT3_CLASS_SILENCE = 0,
    PROJECT3_CLASS_UNKNOWN = 1,
    PROJECT3_CLASS_DOWN = 2,
    PROJECT3_CLASS_GO = 3,
    PROJECT3_CLASS_LEFT = 4,
    PROJECT3_CLASS_NO = 5,
    PROJECT3_CLASS_OFF = 6,
    PROJECT3_CLASS_ON = 7,
    PROJECT3_CLASS_RIGHT = 8,
    PROJECT3_CLASS_ZERO = 9,
    PROJECT3_CLASS_STOP = 10,
    PROJECT3_CLASS_UP = 11,
    PROJECT3_CLASS_YES = 12
} PROJECT3_CLASS_ID;

typedef struct {
    float noise_floor;
    float smooth_energy;
    unsigned short speech_hold_frames;
    unsigned short silence_hold_frames;
    unsigned char active;
} PROJECT3_VAD_STATE;

typedef struct {
    short raw[PROJECT3_RAW_MAX_SAMPLES];
    unsigned int count;
    unsigned char ready;
} PROJECT3_UTTERANCE_BUFFER;

typedef struct {
    unsigned char class_id;
    float confidence;
    unsigned char valid;
} PROJECT3_INFER_RESULT;

typedef struct {
    PROJECT3_APP_STATE app_state;
    PROJECT3_MODEL_STATE model_state;
    PROJECT3_VAD_STATE vad;
    PROJECT3_UTTERANCE_BUFFER utter;
    PROJECT3_INFER_RESULT last_result;
    char main_text[PROJECT3_RESULT_TEXT_LEN];
    char line1[PROJECT3_UI_TEXT_LEN];
    char line2[PROJECT3_UI_TEXT_LEN];
    char last_main_text[PROJECT3_RESULT_TEXT_LEN];
    char last_line1[PROJECT3_UI_TEXT_LEN];
    char last_line2[PROJECT3_UI_TEXT_LEN];
    unsigned long recognized_count;
    unsigned long frame_counter;
    unsigned char pass_through_enable;
    unsigned char input_gate_blocks;
    unsigned char ui_hold_blocks;
    unsigned char redraw_needed;
    unsigned char lcd_refresh_counter;  // 低频刷新计数器
    unsigned char wake_active;          // 唤醒词激活标志: 0=休眠, 1=激活
} PROJECT3_CONTEXT;

// LCD防重入标志
static volatile int s_lcd_busy = 0;

static void Project3_InitContext(PROJECT3_CONTEXT *ctx);
static void Project3_InitUi(PROJECT3_CONTEXT *ctx);
static void Project3_HandleKeys(PROJECT3_CONTEXT *ctx);
static void Project3_HandleTouch(PROJECT3_CONTEXT *ctx);
static void Project3_ProcessAudioBlock(PROJECT3_CONTEXT *ctx, short *block, unsigned int block_samples);
static void Project3_UpdateUi(PROJECT3_CONTEXT *ctx, unsigned char force_redraw);
static void Project3_ModelInit(PROJECT3_CONTEXT *ctx);
static PROJECT3_INFER_RESULT Project3_RunInference(PROJECT3_CONTEXT *ctx, const PROJECT3_UTTERANCE_BUFFER *utter);
static const char *Project3_GetLabel(unsigned char class_id);
static void Project3_ResetUtterance(PROJECT3_CONTEXT *ctx);

#define P3_IDX3(c, h, w, H, W)       ((((c) * (H)) + (h)) * (W) + (w))
#define P3_W4(o, i, kh, kw, IC, KH, KW) (((((o) * (IC)) + (i)) * (KH) + (kh)) * (KW) + (kw))
#define P3_DW(c, kh, kw, KH, KW)     (((c) * (KH) + (kh)) * (KW) + (kw))

static const char *g_project3_labels[PROJECT3_CMD_COUNT] = {
    "_silence_",
    "_unknown_",
    "down",
    "go",
    "left",
    "no",
    "off",
    "on",
    "right",
    "zero",
    "stop",
    "up",
    "yes"
};

#pragma DATA_ALIGN(g_project3_input_block, 8)
static short g_project3_input_block[PROJECT3_BLOCK_SAMPLES];

#pragma DATA_ALIGN(g_project3_output_block, 8)
static short g_project3_output_block[PROJECT3_BLOCK_SAMPLES];

#pragma DATA_ALIGN(g_project3_model_wave, 8)
static float g_project3_model_wave[PROJECT3_MODEL_SAMPLES];

#pragma DATA_ALIGN(g_project3_window, 8)
static float g_project3_window[PROJECT3_WIN_SIZE];

#pragma DATA_ALIGN(g_project3_fft_re, 8)
static float g_project3_fft_re[PROJECT3_FFT_LEN];

#pragma DATA_ALIGN(g_project3_fft_im, 8)
static float g_project3_fft_im[PROJECT3_FFT_LEN];

#pragma DATA_ALIGN(g_project3_spec, 8)
static float g_project3_spec[PROJECT3_FREQ_NUM];

#pragma DATA_ALIGN(g_project3_mel_filter, 8)
static float g_project3_mel_filter[PROJECT3_FREQ_NUM * PROJECT3_MELS_NUM];

#pragma DATA_ALIGN(g_project3_logmel, 8)
static float g_project3_logmel[PROJECT3_MELS_NUM * PROJECT3_MODEL_FRAMES];

#pragma DATA_ALIGN(g_project3_act_a, 8)
static float g_project3_act_a[PROJECT3_ACT_MAX];

#pragma DATA_ALIGN(g_project3_act_b, 8)
static float g_project3_act_b[PROJECT3_ACT_MAX];

#pragma DATA_ALIGN(g_project3_act_c, 8)
static float g_project3_act_c[PROJECT3_ACT_MAX];

#pragma DATA_ALIGN(g_project3_fc_in, 8)
static float g_project3_fc_in[32];

#pragma DATA_ALIGN(g_project3_logits, 8)
static float g_project3_logits[PROJECT3_CMD_COUNT];

#pragma DATA_ALIGN(g_project3_tw_re, 8)
static float g_project3_tw_re[PROJECT3_FFT_LEN / 2];

#pragma DATA_ALIGN(g_project3_tw_im, 8)
static float g_project3_tw_im[PROJECT3_FFT_LEN / 2];

static unsigned char g_project3_tables_ready = 0;

static void Project3_SetAppState(PROJECT3_CONTEXT *ctx, PROJECT3_APP_STATE state);
static void Project3_SetUiText(PROJECT3_CONTEXT *ctx, const char *main_text, const char *line1, const char *line2);
static void Project3_ClearCenterArea(void);
static void Project3_DrawCenterText(const char *text, unsigned long color);
static void Project3_UpdateBottomStatusBar(PROJECT3_CONTEXT *ctx);
static void Project3_DrawHeader(PROJECT3_CONTEXT *ctx);
static void Project3_DrawCardFrame(void);
static void Project3_DrawConfidenceBar(PROJECT3_CONTEXT *ctx);
static void Project3_ClearCardContent(void);
static unsigned long Project3_GetAccentColor(PROJECT3_APP_STATE state);
static void Project3_InitTables(void);
static void Project3_Fft512(float *re, float *im);
static float Project3_PaddedModelSample(int idx);
static void Project3_BuildModelWave(const PROJECT3_UTTERANCE_BUFFER *utter);
static void Project3_ExtractLogMel(const PROJECT3_UTTERANCE_BUFFER *utter, float *out_logmel);
static float Project3_ComputeFrameEnergy(const short *frame, unsigned int len);
static unsigned char Project3_UpdateVad(PROJECT3_CONTEXT *ctx, float frame_energy);
static unsigned char Project3_ShouldEndUtterance(PROJECT3_CONTEXT *ctx);
static void Project3_AppendRawBlock(PROJECT3_CONTEXT *ctx, const short *block, unsigned int block_samples);
static void Project3_CopyInputBlock(short *dst);
static void Project3_FillDacOutput(const PROJECT3_CONTEXT *ctx, const short *src);
static void Project3_WriteOutputBlockToDac(const short *src);
static void Project3_HandleSpeechEnd(PROJECT3_CONTEXT *ctx);
static void Project3_RenderScreen(PROJECT3_CONTEXT *ctx, unsigned char force_redraw);
static void Project3_ServiceUiHold(PROJECT3_CONTEXT *ctx);

static void Project3_Conv2d(const float *restrict in, float *restrict out,
                            int in_c, int in_h, int in_w,
                            int out_c, int out_h, int out_w,
                            int k_h, int k_w, int s_h, int s_w,
                            int p_h, int p_w, const float *restrict weight);
static void Project3_DepthwiseConv2d(const float *restrict in, float *restrict out,
                                     int channels, int in_h, int in_w,
                                     int out_h, int out_w,
                                     int k_h, int k_w, int s_h, int s_w,
                                     int p_h, int p_w, const float *restrict weight);
static void Project3_ScaleBiasReLU(float *restrict x, int channels, int h, int w,
                                   int bn_idx, unsigned char do_relu);
static void Project3_InitMergedBN(void);
static void Project3_AddRelu(float *x, const float *identity, int count);
static void Project3_BCResBlock(const float *in, float *out,
                                int in_c, int in_h, int in_w,
                                int out_c, int out_h, int out_w,
                                int stride_h, int stride_w,
                                const float *conv1_w,
                                int bn1_idx,
                                const float *dw_w,
                                int bn2_idx,
                                const float *conv2_w,
                                int bn3_idx,
                                const float *shortcut_w,
                                int sc_bn_idx);
static void Project3_BCResNetForward(const float *logmel, float *logits);
static PROJECT3_INFER_RESULT Project3_LogitsToResult(const float *logits);
static unsigned char Project3_AcceptResult(PROJECT3_INFER_RESULT *result, const PROJECT3_UTTERANCE_BUFFER *utter);

static void Project3_SetAppState(PROJECT3_CONTEXT *ctx, PROJECT3_APP_STATE state)
{
    if (ctx->app_state != state) {
        ctx->app_state = state;
        ctx->redraw_needed = 1;
    }
}

static void Project3_SetUiText(PROJECT3_CONTEXT *ctx, const char *main_text, const char *line1, const char *line2)
{
    if (main_text != NULL) {
        strncpy(ctx->main_text, main_text, PROJECT3_RESULT_TEXT_LEN - 1);
        ctx->main_text[PROJECT3_RESULT_TEXT_LEN - 1] = '\0';
    }
    if (line1 != NULL) {
        strncpy(ctx->line1, line1, PROJECT3_UI_TEXT_LEN - 1);
        ctx->line1[PROJECT3_UI_TEXT_LEN - 1] = '\0';
    }
    if (line2 != NULL) {
        strncpy(ctx->line2, line2, PROJECT3_UI_TEXT_LEN - 1);
        ctx->line2[PROJECT3_UI_TEXT_LEN - 1] = '\0';
    }
    ctx->redraw_needed = 1;
}

static void Project3_ClearAndDrawText(const char *text, unsigned long color)
{
    tRectangle rect;

    // 使用固定布局宏
    rect.sXMin = P3_CLAMP_X(P3_TEXT_AREA_X);
    rect.sYMin = P3_CLAMP_Y(P3_TEXT_AREA_Y);
    rect.sXMax = P3_CLAMP_X(P3_TEXT_AREA_X + P3_TEXT_AREA_W - 1);
    rect.sYMax = P3_CLAMP_Y(P3_TEXT_AREA_Y + P3_TEXT_AREA_H - 1);

    // 第1步：清空区域为纯黑
    GrContextForegroundSet(&Lcd_Context, ClrBlack);
    GrRectFill(&Lcd_Context, &rect);

    // 第2步：立即在同一区域绘制文字（不设置裁剪区域，避免状态切换）
    GrContextForegroundSet(&Lcd_Context, color);
    GrContextBackgroundSet(&Lcd_Context, ClrBlack);
    GrContextFontSet(&Lcd_Context, &g_sFontCm48);
    GrStringDraw(&Lcd_Context, text, -1, P3_TEXT_X, P3_TEXT_Y, 1);
}

static void Project3_UpdateBottomStatusBar(PROJECT3_CONTEXT *ctx)
{
    tRectangle bar_rect;
    int text_w;

    /* 清除底部状态栏区域 */
    bar_rect.sXMin = 0;
    bar_rect.sYMin = P3_BOTTOM_BAR_Y;
    bar_rect.sXMax = P3_LCD_W - 1;
    bar_rect.sYMax = P3_LCD_H - 1;
    GrContextForegroundSet(&Lcd_Context, ClrBlack);
    GrRectFill(&Lcd_Context, &bar_rect);

    /* 底部状态栏分隔线 */
    GrContextForegroundSet(&Lcd_Context, ClrDimGray);
    GrLineDrawH(&Lcd_Context, 0, P3_LCD_W - 1, P3_BOTTOM_BAR_Y);

    /* 状态栏文字 - 第一行 */
    GrContextForegroundSet(&Lcd_Context, ClrLightSteelBlue);
    GrContextBackgroundSet(&Lcd_Context, ClrBlack);
    GrContextFontSet(&Lcd_Context, &g_sFontCm16);
    text_w = GrStringWidthGet(&Lcd_Context, ctx->line1, -1);
    if (text_w > P3_LCD_W - 20) text_w = P3_LCD_W - 20;
    GrStringDraw(&Lcd_Context, ctx->line1, -1,
                 (P3_LCD_W - text_w) / 2, P3_BOTTOM_BAR_Y + 2, 1);

    /* 状态栏文字 - 第二行 */
    text_w = GrStringWidthGet(&Lcd_Context, ctx->line2, -1);
    if (text_w > P3_LCD_W - 20) text_w = P3_LCD_W - 20;
    GrStringDraw(&Lcd_Context, ctx->line2, -1,
                 (P3_LCD_W - text_w) / 2, P3_BOTTOM_BAR_Y + 20, 1);
}

static unsigned long Project3_GetAccentColor(PROJECT3_APP_STATE state)
{
    switch (state) {
        case PROJECT3_APP_BOOT:         return ClrGray;
        case PROJECT3_APP_LISTENING:    return ClrLime;
        case PROJECT3_APP_SPEECH:       return ClrDarkOrange;
        case PROJECT3_APP_INFERENCING:  return ClrCyan;
        case PROJECT3_APP_RESULT:       return ClrWhite;
        default:                        return ClrLightSteelBlue;
    }
}

static void Project3_DrawHeader(PROJECT3_CONTEXT *ctx)
{
    tRectangle header_rect;
    unsigned long accent = Project3_GetAccentColor(ctx->app_state);
    char status_text[16];

    /* 标题栏背景 */
    header_rect.sXMin = 0;
    header_rect.sYMin = P3_HEADER_Y;
    header_rect.sXMax = P3_LCD_W - 1;
    header_rect.sYMax = P3_HEADER_Y + P3_HEADER_H - 1;
    GrContextForegroundSet(&Lcd_Context, P3_HEADER_BG);
    GrRectFill(&Lcd_Context, &header_rect);

    /* 底部分割线 - 强调色 */
    GrContextForegroundSet(&Lcd_Context, accent);
    GrLineDrawH(&Lcd_Context, 0, P3_LCD_W - 1, P3_HEADER_SEPARATOR_Y);

    /* 标题文字 */
    GrContextForegroundSet(&Lcd_Context, ClrWhite);
    GrContextBackgroundSet(&Lcd_Context, P3_HEADER_BG);
    GrContextFontSet(&Lcd_Context, &P3_TITLE_FONT);
    GrStringDraw(&Lcd_Context, "Voice Command Recognition", -1, P3_TITLE_X, P3_TITLE_Y, 1);

    /* 右侧状态文字 */
    GrContextFontSet(&Lcd_Context, &g_sFontCm18);
    switch (ctx->app_state) {
        case PROJECT3_APP_BOOT:         strcpy(status_text, "BOOT"); break;
        case PROJECT3_APP_LISTENING:    strcpy(status_text, "READY"); break;
        case PROJECT3_APP_SPEECH:       strcpy(status_text, "REC"); break;
        case PROJECT3_APP_INFERENCING:  strcpy(status_text, "PROC"); break;
        case PROJECT3_APP_RESULT:       strcpy(status_text, "DONE"); break;
        default:                        strcpy(status_text, ""); break;
    }
    GrStringDraw(&Lcd_Context, status_text, -1, P3_STATUS_DOT_X - 55, P3_TITLE_Y + 4, 1);

    /* 状态指示灯 - 实心圆 + 外环 */
    GrContextForegroundSet(&Lcd_Context, accent);
    GrCircleFill(&Lcd_Context, P3_STATUS_DOT_X, P3_STATUS_DOT_Y, P3_STATUS_DOT_R);
    GrContextForegroundSet(&Lcd_Context, ClrWhite);
    GrCircleDraw(&Lcd_Context, P3_STATUS_DOT_X, P3_STATUS_DOT_Y, P3_STATUS_DOT_RING_R);

}

static void Project3_DrawCardFrame(void)
{
    tRectangle card_border;
    int i;

    /* 卡片阴影层（通过绘制多层偏移矩形模拟） */
    card_border.sXMin = P3_CARD_X + 3;
    card_border.sYMin = P3_CARD_Y + 3;
    card_border.sXMax = P3_CARD_X + P3_CARD_W - 1 + 3;
    card_border.sYMax = P3_CARD_Y + P3_CARD_H - 1 + 3;
    GrContextForegroundSet(&Lcd_Context, ClrBlack);
    GrRectFill(&Lcd_Context, &card_border);

    /* 卡片主体背景 */
    card_border.sXMin = P3_CARD_X;
    card_border.sYMin = P3_CARD_Y;
    card_border.sXMax = P3_CARD_X + P3_CARD_W - 1;
    card_border.sYMax = P3_CARD_Y + P3_CARD_H - 1;
    GrContextForegroundSet(&Lcd_Context, ClrBlack);
    GrRectFill(&Lcd_Context, &card_border);

    /* 卡片边框（细线） */
    GrContextForegroundSet(&Lcd_Context, ClrDimGray);
    GrRectDraw(&Lcd_Context, &card_border);

    /* 左上角和右上角装饰短线 */
    GrContextForegroundSet(&Lcd_Context, ClrLightSteelBlue);
    for (i = 0; i < 20; i++) {
        /* 左上角水平 */
        GrLineDrawH(&Lcd_Context, P3_CARD_X + 4, P3_CARD_X + 4 + i, P3_CARD_Y + 1);
        /* 左上角垂直 */
        GrLineDrawV(&Lcd_Context, P3_CARD_X + 1, P3_CARD_Y + 4, P3_CARD_Y + 4 + i);
        /* 右上角水平 */
        GrLineDrawH(&Lcd_Context, P3_CARD_X + P3_CARD_W - 5 - i, P3_CARD_X + P3_CARD_W - 5, P3_CARD_Y + 1);
        /* 右上角垂直 */
        GrLineDrawV(&Lcd_Context, P3_CARD_X + P3_CARD_W - 2, P3_CARD_Y + 4, P3_CARD_Y + 4 + i);
    }
}

static void Project3_ClearCardContent(void)
{
    tRectangle inner;

    /* 清除卡片内部区域（保留边框） */
    inner.sXMin = P3_CARD_X + 2;
    inner.sYMin = P3_CARD_Y + 2;
    inner.sXMax = P3_CARD_X + P3_CARD_W - 3;
    inner.sYMax = P3_CARD_Y + P3_CARD_H - 3;
    GrContextForegroundSet(&Lcd_Context, ClrBlack);
    GrRectFill(&Lcd_Context, &inner);
}

static void Project3_DrawConfidenceBar(PROJECT3_CONTEXT *ctx)
{
    tRectangle bar_bg, bar_fill;
    float confidence = ctx->last_result.confidence;
    int fill_w;
    char conf_str[24];

    if (ctx->app_state != PROJECT3_APP_RESULT) return;
    if (confidence <= 0.0f) return;

    /* 置信度文字 */
    sprintf(conf_str, "Confidence: %.0f%%", confidence * 100.0f);
    GrContextForegroundSet(&Lcd_Context, ClrLightSteelBlue);
    GrContextBackgroundSet(&Lcd_Context, ClrBlack);
    GrContextFontSet(&Lcd_Context, &g_sFontCm22);
    GrStringDraw(&Lcd_Context, conf_str, -1,
                 (P3_LCD_W - GrStringWidthGet(&Lcd_Context, conf_str, -1)) / 2,
                 P3_CONFIDENCE_TEXT_Y, 1);

    /* 进度条背景 */
    bar_bg.sXMin = P3_CONFIDENCE_BAR_X;
    bar_bg.sYMin = P3_CONFIDENCE_BAR_Y;
    bar_bg.sXMax = P3_CONFIDENCE_BAR_X + P3_CONFIDENCE_BAR_W - 1;
    bar_bg.sYMax = P3_CONFIDENCE_BAR_Y + P3_CONFIDENCE_BAR_H - 1;
    GrContextForegroundSet(&Lcd_Context, ClrDarkSlateGray);
    GrRectFill(&Lcd_Context, &bar_bg);

    /* 进度条填充 */
    fill_w = (int)(P3_CONFIDENCE_BAR_W * confidence);
    if (fill_w < 4) fill_w = 4;
    if (fill_w > P3_CONFIDENCE_BAR_W - 4) fill_w = P3_CONFIDENCE_BAR_W - 4;

    bar_fill.sXMin = P3_CONFIDENCE_BAR_X + 2;
    bar_fill.sYMin = P3_CONFIDENCE_BAR_Y + 2;
    bar_fill.sXMax = P3_CONFIDENCE_BAR_X + fill_w - 1;
    bar_fill.sYMax = P3_CONFIDENCE_BAR_Y + P3_CONFIDENCE_BAR_H - 3;
    GrContextForegroundSet(&Lcd_Context, Project3_GetAccentColor(ctx->app_state));
    GrRectFill(&Lcd_Context, &bar_fill);
}

static void Project3_InitTables(void)
{
    int i, j;
    float all_freq;
    float m_min;
    float m_max;
    float m_pts[PROJECT3_MELS_NUM + 2];
    float f_pts[PROJECT3_MELS_NUM + 2];
    float f_diff[PROJECT3_MELS_NUM + 1];

    if (g_project3_tables_ready) {
        return;
    }

    for (i = 0; i < PROJECT3_WIN_SIZE; i++) {
        g_project3_window[i] = 0.5f * (1.0f - cosf(2.0f * PROJECT3_PI * (float)i / (float)PROJECT3_WIN_SIZE));
    }

    for (i = 0; i < PROJECT3_FFT_LEN / 2; i++) {
        float angle = 2.0f * PROJECT3_PI * (float)i / (float)PROJECT3_FFT_LEN;
        g_project3_tw_re[i] = cosf(angle);
        g_project3_tw_im[i] = -sinf(angle);
    }

    m_min = 2595.0f * log10f(1.0f);
    m_max = 2595.0f * log10f(1.0f + ((float)PROJECT3_MODEL_SAMPLE_RATE / 2.0f) / 700.0f);

    for (i = 0; i < PROJECT3_MELS_NUM + 2; i++) {
        m_pts[i] = m_min + (m_max - m_min) * (float)i / (float)(PROJECT3_MELS_NUM + 1);
        f_pts[i] = 700.0f * (powf(10.0f, m_pts[i] / 2595.0f) - 1.0f);
    }

    for (i = 0; i < PROJECT3_MELS_NUM + 1; i++) {
        f_diff[i] = f_pts[i + 1] - f_pts[i];
    }

    for (i = 0; i < PROJECT3_FREQ_NUM; i++) {
        all_freq = ((float)i) * ((float)PROJECT3_MODEL_SAMPLE_RATE / 2.0f) / (float)(PROJECT3_FREQ_NUM - 1);
        for (j = 0; j < PROJECT3_MELS_NUM; j++) {
            float slope_j = f_pts[j] - all_freq;
            float slope_j2 = f_pts[j + 2] - all_freq;
            float left = -slope_j / f_diff[j];
            float right = slope_j2 / f_diff[j + 1];
            float val = (left < right) ? left : right;
            if (val < 0.0f) val = 0.0f;
            g_project3_mel_filter[i * PROJECT3_MELS_NUM + j] = val;
        }
    }

    g_project3_tables_ready = 1;
}

static void Project3_Fft512(float *re, float *im)
{
    int i, j, bit, len, half, step, base;

    j = 0;
    for (i = 1; i < PROJECT3_FFT_LEN; i++) {
        bit = PROJECT3_FFT_LEN >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            float tr = re[i];
            float ti = im[i];
            re[i] = re[j];
            im[i] = im[j];
            re[j] = tr;
            im[j] = ti;
        }
    }

    for (len = 2; len <= PROJECT3_FFT_LEN; len <<= 1) {
        half = len >> 1;
        step = PROJECT3_FFT_LEN / len;
        for (base = 0; base < PROJECT3_FFT_LEN; base += len) {
            for (j = 0; j < half; j++) {
                int even = base + j;
                int odd = even + half;
                int tw = j * step;
                float wr = g_project3_tw_re[tw];
                float wi = g_project3_tw_im[tw];
                float tr = wr * re[odd] - wi * im[odd];
                float ti = wr * im[odd] + wi * re[odd];
                float ur = re[even];
                float ui = im[even];
                re[even] = ur + tr;
                im[even] = ui + ti;
                re[odd] = ur - tr;
                im[odd] = ui - ti;
            }
        }
    }

}

static float Project3_PaddedModelSample(int idx)
{
    if (idx < PROJECT3_WIN_SIZE / 2) {
        int src = PROJECT3_WIN_SIZE / 2 - idx;
        if (src >= PROJECT3_MODEL_SAMPLES) src = PROJECT3_MODEL_SAMPLES - 1;
        return g_project3_model_wave[src];
    }
    if (idx < PROJECT3_WIN_SIZE / 2 + PROJECT3_MODEL_SAMPLES) {
        return g_project3_model_wave[idx - PROJECT3_WIN_SIZE / 2];
    }
    {
        int tail = idx - (PROJECT3_WIN_SIZE / 2 + PROJECT3_MODEL_SAMPLES);
        int src = PROJECT3_MODEL_SAMPLES - 2 - tail;
        if (src < 0) src = 0;
        return g_project3_model_wave[src];
    }
}

static void Project3_BuildModelWave(const PROJECT3_UTTERANCE_BUFFER *utter)
{
    int n;
    for (n = 0; n < PROJECT3_MODEL_SAMPLES; n++) {
        int q = n * 5;
        int base = q >> 2;
        int rem = q & 3;
        float frac = (float)rem * 0.25f;
        float sample = 0.0f;

        if ((unsigned int)(base + 1) < utter->count) {
            sample = (1.0f - frac) * (float)utter->raw[base] + frac * (float)utter->raw[base + 1];
        } else if ((unsigned int)base < utter->count) {
            sample = (float)utter->raw[base];
        }
        g_project3_model_wave[n] = sample * (PROJECT3_INPUT_GAIN / 32768.0f);
    }
}

static void Project3_ExtractLogMel(const PROJECT3_UTTERANCE_BUFFER *utter, float *out_logmel)
{
    int frame, i, m, f;

    Project3_InitTables();
    Project3_BuildModelWave(utter);

    for (frame = 0; frame < PROJECT3_MODEL_FRAMES; frame++) {
        int start = frame * PROJECT3_HOP_SIZE;
        for (i = 0; i < PROJECT3_FFT_LEN; i++) {
            g_project3_fft_re[i] = 0.0f;
            g_project3_fft_im[i] = 0.0f;
        }
        for (i = 0; i < PROJECT3_WIN_SIZE; i++) {
            g_project3_fft_re[i] = Project3_PaddedModelSample(start + i) * g_project3_window[i];
        }

        Project3_Fft512(g_project3_fft_re, g_project3_fft_im);

        for (f = 0; f < PROJECT3_FREQ_NUM; f++) {
            g_project3_spec[f] = g_project3_fft_re[f] * g_project3_fft_re[f]
                                + g_project3_fft_im[f] * g_project3_fft_im[f];
        }

        for (m = 0; m < PROJECT3_MELS_NUM; m++) {
            float mel_energy = 1.0e-6f;
            for (f = 0; f < PROJECT3_FREQ_NUM; f++) {
                mel_energy += g_project3_spec[f] * g_project3_mel_filter[f * PROJECT3_MELS_NUM + m];
            }
            out_logmel[m * PROJECT3_MODEL_FRAMES + frame] = logf(mel_energy);
        }
    }
}

static float Project3_ComputeFrameEnergy(const short *frame, unsigned int len)
{
    unsigned int i;
    float mean = 0.0f;
    float energy = 0.0f;

    for (i = 0; i < len; i++) {
        mean += (float)frame[i];
    }
    mean /= (float)len;

    for (i = 0; i < len; i++) {
        float sample = ((float)frame[i] - mean) / 32768.0f;
        energy += sample * sample;
    }

    return energy / (float)len;
}

static unsigned char Project3_UpdateVad(PROJECT3_CONTEXT *ctx, float frame_energy)
{
    PROJECT3_VAD_STATE *vad = &ctx->vad;
    const float floor_alpha = 0.995f;
    const float smooth_alpha = 0.90f;
    const float start_ratio = 4.5f;
    const float stop_ratio = 1.3f;
    const unsigned short start_frames = 4;

    if (vad->noise_floor <= 0.0f) {
        vad->noise_floor = frame_energy;
    }

    vad->smooth_energy = smooth_alpha * vad->smooth_energy + (1.0f - smooth_alpha) * frame_energy;

    if (!vad->active) {
        vad->noise_floor = floor_alpha * vad->noise_floor + (1.0f - floor_alpha) * vad->smooth_energy;
    }

    if (vad->smooth_energy > vad->noise_floor * start_ratio) {
        if (vad->speech_hold_frames < 255) vad->speech_hold_frames++;
    } else {
        vad->speech_hold_frames = 0;
    }

    if (vad->active) {
        if (vad->smooth_energy < vad->noise_floor * stop_ratio) {
            if (vad->silence_hold_frames < 255) vad->silence_hold_frames++;
        } else {
            vad->silence_hold_frames = 0;
        }
    }

    if (!vad->active && vad->speech_hold_frames >= start_frames) {
        vad->active = 1;
        vad->speech_hold_frames = 0;
        vad->silence_hold_frames = 0;
        return 1;
    }

    return 0;
}

static unsigned char Project3_ShouldEndUtterance(PROJECT3_CONTEXT *ctx)
{
    const unsigned short stop_frames = 15;
    const unsigned int min_samples = PROJECT3_MIN_UTTERANCE_SAMPLES;

    if (ctx->vad.active && ctx->vad.silence_hold_frames >= stop_frames) {
        ctx->vad.active = 0;
        ctx->vad.silence_hold_frames = 0;
        if (ctx->utter.count >= min_samples) {
            ctx->utter.ready = 1;
            return 1;
        }
        Project3_ResetUtterance(ctx);
    }

    if (ctx->utter.count >= PROJECT3_RAW_MAX_SAMPLES) {
        ctx->vad.active = 0;
        ctx->utter.ready = 1;
        return 1;
    }

    return 0;
}

static void Project3_AppendRawBlock(PROJECT3_CONTEXT *ctx, const short *block, unsigned int block_samples)
{
    unsigned int i;
    for (i = 0; i < block_samples && ctx->utter.count < PROJECT3_RAW_MAX_SAMPLES; i++) {
        ctx->utter.raw[ctx->utter.count++] = block[i];
    }
}

static void Project3_CopyInputBlock(short *dst)
{
    short *src = (AD_Ping_Pong == AD_BUFFER_PONG) ? AD_CH1_Buf0 : AD_CH1_Buf1;
    memcpy(dst, src, sizeof(short) * PROJECT3_BLOCK_SAMPLES);
}

static void Project3_FillDacOutput(const PROJECT3_CONTEXT *ctx, const short *src)
{
    unsigned int i;
    if (ctx->pass_through_enable) {
        memcpy(g_project3_output_block, src, sizeof(short) * PROJECT3_BLOCK_SAMPLES);
    } else {
        for (i = 0; i < PROJECT3_BLOCK_SAMPLES; i++) {
            g_project3_output_block[i] = 0;
        }
    }
}

static void Project3_WriteOutputBlockToDac(const short *src)
{
    if (DA_Ping_Pong == DA_BUFFER_PONG) {
        memcpy(DA_CH1_Buf0, src, sizeof(short) * PROJECT3_BLOCK_SAMPLES);
    } else {
        memcpy(DA_CH1_Buf1, src, sizeof(short) * PROJECT3_BLOCK_SAMPLES);
    }
}

/* ============================================================
 *  Project3_InitMergedBN — 预计算合并 BN 参数 (启动时调用一次)
 *  消除推理时每条通道的 sqrt/div, 约 15-20% 加速
 * ============================================================ */
static void Project3_InitMergedBN(void)
{
    if (g_project3_merged_bn_ready) return;
    int c;
    float eps = PROJECT3_BN_EPS;

    /* 辅助宏: 对每个 BN 层计算 merged_w, merged_b */
#define P3_MERGE_BN(idx, gamma, beta, mean, var, ch) do { \
    for (c = 0; c < (ch); c++) { \
        float mw = (gamma)[c] / sqrtf((var)[c] + eps); \
        g_project3_merged_bn_w[idx][c] = mw; \
        g_project3_merged_bn_b[idx][c] = (beta)[c] - (mean)[c] * mw; \
    } \
} while(0)

    P3_MERGE_BN(P3_BN_IDX_conv1, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, 16);
    P3_MERGE_BN(P3_BN_IDX_l1b1, layer1_bn1_weight, layer1_bn1_bias, layer1_bn1_running_mean, layer1_bn1_running_var, 8);
    P3_MERGE_BN(P3_BN_IDX_l1b2, layer1_bn2_weight, layer1_bn2_bias, layer1_bn2_running_mean, layer1_bn2_running_var, 8);
    P3_MERGE_BN(P3_BN_IDX_l1b3, layer1_bn3_weight, layer1_bn3_bias, layer1_bn3_running_mean, layer1_bn3_running_var, 8);
    P3_MERGE_BN(P3_BN_IDX_l1sc, layer1_shortcut_1_weight, layer1_shortcut_1_bias, layer1_shortcut_1_running_mean, layer1_shortcut_1_running_var, 8);
    P3_MERGE_BN(P3_BN_IDX_l2b1, layer2_bn1_weight, layer2_bn1_bias, layer2_bn1_running_mean, layer2_bn1_running_var, 12);
    P3_MERGE_BN(P3_BN_IDX_l2b2, layer2_bn2_weight, layer2_bn2_bias, layer2_bn2_running_mean, layer2_bn2_running_var, 12);
    P3_MERGE_BN(P3_BN_IDX_l2b3, layer2_bn3_weight, layer2_bn3_bias, layer2_bn3_running_mean, layer2_bn3_running_var, 12);
    P3_MERGE_BN(P3_BN_IDX_l2sc, layer2_shortcut_1_weight, layer2_shortcut_1_bias, layer2_shortcut_1_running_mean, layer2_shortcut_1_running_var, 12);
    P3_MERGE_BN(P3_BN_IDX_l3b1, layer3_bn1_weight, layer3_bn1_bias, layer3_bn1_running_mean, layer3_bn1_running_var, 16);
    P3_MERGE_BN(P3_BN_IDX_l3b2, layer3_bn2_weight, layer3_bn2_bias, layer3_bn2_running_mean, layer3_bn2_running_var, 16);
    P3_MERGE_BN(P3_BN_IDX_l3b3, layer3_bn3_weight, layer3_bn3_bias, layer3_bn3_running_mean, layer3_bn3_running_var, 16);
    P3_MERGE_BN(P3_BN_IDX_l3sc, layer3_shortcut_1_weight, layer3_shortcut_1_bias, layer3_shortcut_1_running_mean, layer3_shortcut_1_running_var, 16);
    P3_MERGE_BN(P3_BN_IDX_dw,   bn_dw_weight, bn_dw_bias, bn_dw_running_mean, bn_dw_running_var, 16);
    P3_MERGE_BN(P3_BN_IDX_pw,   bn_pw_weight, bn_pw_bias, bn_pw_running_mean, bn_pw_running_var, 20);
    P3_MERGE_BN(P3_BN_IDX_conv2, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var, 20);
    P3_MERGE_BN(P3_BN_IDX_expand, bn_expand_weight, bn_expand_bias, bn_expand_running_mean, bn_expand_running_var, 32);

#undef P3_MERGE_BN
    g_project3_merged_bn_ready = 1;
}

/* ============================================================
 *  Project3_ScaleBiasReLU — 替代 BatchNorm (使用预合并参数)
 *  y = x * merged_w[c] + merged_b[c]; if (do_relu && y < 0) y = 0
 *  比原 BatchNorm 快 3-5x (无 sqrt/div, 无逐通道重复计算)
 * ============================================================ */
static void Project3_ScaleBiasReLU(float *restrict x, int channels, int h, int w,
                                   int bn_idx, unsigned char do_relu)
{
    int c, i, hw = h * w;
    const float *restrict mw = g_project3_merged_bn_w[bn_idx];
    const float *restrict mb = g_project3_merged_bn_b[bn_idx];
    float *restrict ptr;

    for (c = 0; c < channels; c++) {
        float scale = mw[c];
        float bias = mb[c];
        ptr = &x[c * hw];
        if (do_relu) {
            #pragma MUST_ITERATE(1, , 4)
            for (i = 0; i < hw; i++) {
                float v = ptr[i] * scale + bias;
                if (v < 0.0f) v = 0.0f;
                ptr[i] = v;
            }
        } else {
            #pragma MUST_ITERATE(1, , 4)
            for (i = 0; i < hw; i++) {
                ptr[i] = ptr[i] * scale + bias;
            }
        }
    }
}

/* ============================================================
 *  Project3_Conv2d (优化版) — restrict + 循环展开 + pragma
 * ============================================================ */
static void Project3_Conv2d(const float *restrict in, float *restrict out,
                            int in_c, int in_h, int in_w,
                            int out_c, int out_h, int out_w,
                            int k_h, int k_w, int s_h, int s_w,
                            int p_h, int p_w, const float *restrict weight)
{
    int oc, oh, ow, ic, kh, kw;
    int in_hw = in_h * in_w;
    int out_hw = out_h * out_w;
    int wt_plane = in_c * k_h * k_w;

    #pragma MUST_ITERATE(1, , 4)
    for (oc = 0; oc < out_c; oc++) {
        const float *restrict wt_oc = &weight[oc * wt_plane];
        #pragma MUST_ITERATE(1, , 4)
        for (oh = 0; oh < out_h; oh++) {
            #pragma MUST_ITERATE(1, , 4)
            for (ow = 0; ow < out_w; ow++) {
                float sum = 0.0f;
                #pragma MUST_ITERATE(1, , 4)
                for (ic = 0; ic < in_c; ic++) {
                    const float *restrict in_ic = &in[ic * in_hw];
                    const float *restrict wt_ic = &wt_oc[ic * k_h * k_w];
                    for (kh = 0; kh < k_h; kh++) {
                        int ih = oh * s_h + kh - p_h;
                        if ((unsigned int)ih >= (unsigned int)in_h) continue;
                        const float *restrict in_row = &in_ic[ih * in_w];
                        /* 手动展开 k_w 循环以利用 C6748 VLIW */
                        switch (k_w) {
                            case 1:
                                { int iw = ow * s_w - p_w;
                                  if ((unsigned int)iw < (unsigned int)in_w)
                                      sum += in_row[iw] * wt_ic[kh * k_w]; }
                                break;
                            case 3:
                                { int iw0 = ow * s_w + 0 - p_w;
                                  if ((unsigned int)iw0 < (unsigned int)in_w)
                                      sum += in_row[iw0] * wt_ic[kh * k_w + 0];
                                  int iw1 = ow * s_w + 1 - p_w;
                                  if ((unsigned int)iw1 < (unsigned int)in_w)
                                      sum += in_row[iw1] * wt_ic[kh * k_w + 1];
                                  int iw2 = ow * s_w + 2 - p_w;
                                  if ((unsigned int)iw2 < (unsigned int)in_w)
                                      sum += in_row[iw2] * wt_ic[kh * k_w + 2]; }
                                break;
                            case 5:
                                for (kw = 0; kw < 5; kw++) {
                                    int iw = ow * s_w + kw - p_w;
                                    if ((unsigned int)iw < (unsigned int)in_w)
                                        sum += in_row[iw] * wt_ic[kh * k_w + kw];
                                }
                                break;
                            default:
                                for (kw = 0; kw < k_w; kw++) {
                                    int iw = ow * s_w + kw - p_w;
                                    if ((unsigned int)iw < (unsigned int)in_w)
                                        sum += in_row[iw] * wt_ic[kh * k_w + kw];
                                }
                                break;
                        }
                    }
                }
                out[oc * out_hw + oh * out_w + ow] = sum;
            }
        }
    }
}

/* ============================================================
 *  Project3_DepthwiseConv2d (优化版)
 * ============================================================ */
static void Project3_DepthwiseConv2d(const float *restrict in, float *restrict out,
                                     int channels, int in_h, int in_w,
                                     int out_h, int out_w,
                                     int k_h, int k_w, int s_h, int s_w,
                                     int p_h, int p_w, const float *restrict weight)
{
    int c, oh, ow, kh, kw;
    int in_hw = in_h * in_w;
    int out_hw = out_h * out_w;

    #pragma MUST_ITERATE(1, , 4)
    for (c = 0; c < channels; c++) {
        const float *restrict in_c = &in[c * in_hw];
        const float *restrict wt_c = &weight[c * k_h * k_w];
        float *restrict out_c = &out[c * out_hw];
        #pragma MUST_ITERATE(1, , 4)
        for (oh = 0; oh < out_h; oh++) {
            #pragma MUST_ITERATE(1, , 4)
            for (ow = 0; ow < out_w; ow++) {
                float sum = 0.0f;
                for (kh = 0; kh < k_h; kh++) {
                    int ih = oh * s_h + kh - p_h;
                    if ((unsigned int)ih >= (unsigned int)in_h) continue;
                    const float *restrict in_row = &in_c[ih * in_w];
                    switch (k_w) {
                        case 3:
                            { int iw0 = ow * s_w - p_w;
                              if ((unsigned int)iw0 < (unsigned int)in_w)
                                  sum += in_row[iw0] * wt_c[kh * 3 + 0];
                              int iw1 = ow * s_w + 1 - p_w;
                              if ((unsigned int)iw1 < (unsigned int)in_w)
                                  sum += in_row[iw1] * wt_c[kh * 3 + 1];
                              int iw2 = ow * s_w + 2 - p_w;
                              if ((unsigned int)iw2 < (unsigned int)in_w)
                                  sum += in_row[iw2] * wt_c[kh * 3 + 2]; }
                            break;
                        default:
                            for (kw = 0; kw < k_w; kw++) {
                                int iw = ow * s_w + kw - p_w;
                                if ((unsigned int)iw < (unsigned int)in_w)
                                    sum += in_row[iw] * wt_c[kh * k_w + kw];
                            }
                            break;
                    }
                }
                out_c[oh * out_w + ow] = sum;
            }
        }
    }
}

/* Project3_BatchNorm 已被 Project3_ScaleBiasReLU 替代 (使用预合并 BN 参数, 3-5x 加速) */

static void Project3_AddRelu(float *restrict x, const float *restrict identity, int count)
{
    int i;
    #pragma MUST_ITERATE(1, , 4)
    for (i = 0; i < count; i++) {
        float v = x[i] + identity[i];
        x[i] = (v < 0.0f) ? 0.0f : v;
    }
}

static void Project3_BCResBlock(const float *in, float *out,
                                int in_c, int in_h, int in_w,
                                int out_c, int out_h, int out_w,
                                int stride_h, int stride_w,
                                const float *conv1_w,
                                int bn1_idx,
                                const float *dw_w,
                                int bn2_idx,
                                const float *conv2_w,
                                int bn3_idx,
                                const float *shortcut_w,
                                int sc_bn_idx)
{
    /* expand: 1x1 Conv + BN + ReLU */
    Project3_Conv2d(in, out, in_c, in_h, in_w, out_c, in_h, in_w, 1, 1, 1, 1, 0, 0, conv1_w);
    Project3_ScaleBiasReLU(out, out_c, in_h, in_w, bn1_idx, 1);

    /* depthwise: 3x3 DW Conv + BN + ReLU */
    Project3_DepthwiseConv2d(out, g_project3_act_c, out_c, in_h, in_w, out_h, out_w, 3, 3, stride_h, stride_w, 1, 1, dw_w);
    Project3_ScaleBiasReLU(g_project3_act_c, out_c, out_h, out_w, bn2_idx, 1);

    /* project: 1x1 Conv + BN (no ReLU) */
    Project3_Conv2d(g_project3_act_c, out, out_c, out_h, out_w, out_c, out_h, out_w, 1, 1, 1, 1, 0, 0, conv2_w);
    Project3_ScaleBiasReLU(out, out_c, out_h, out_w, bn3_idx, 0);

    /* shortcut: 1x1 Conv + BN (no ReLU) */
    Project3_Conv2d(in, g_project3_act_c, in_c, in_h, in_w, out_c, out_h, out_w, 1, 1, stride_h, stride_w, 0, 0, shortcut_w);
    Project3_ScaleBiasReLU(g_project3_act_c, out_c, out_h, out_w, sc_bn_idx, 0);

    /* add + ReLU */
    Project3_AddRelu(out, g_project3_act_c, out_c * out_h * out_w);
}

static void Project3_BCResNetForward(const float *logmel, float *logits)
{
    int c, w, cls, i;

    /* 确保 BN 参数已合并 (首次调用时执行) */
    if (!g_project3_merged_bn_ready) Project3_InitMergedBN();

    /* conv1: 1→16, 3×3, stride=(2,1) + BN + ReLU */
    Project3_Conv2d(logmel, g_project3_act_a, 1, 40, PROJECT3_MODEL_FRAMES,
                    16, 20, PROJECT3_MODEL_FRAMES, 3, 3, 2, 1, 1, 1, conv1_weight);
    Project3_ScaleBiasReLU(g_project3_act_a, 16, 20, PROJECT3_MODEL_FRAMES, P3_BN_IDX_conv1, 1);

    /* block1: 16→8, stride=(1,1) */
    Project3_BCResBlock(g_project3_act_a, g_project3_act_b,
                        16, 20, PROJECT3_MODEL_FRAMES, 8, 20, PROJECT3_MODEL_FRAMES, 1, 1,
                        layer1_conv1_weight, P3_BN_IDX_l1b1,
                        layer1_dwconv_weight, P3_BN_IDX_l1b2,
                        layer1_conv2_weight, P3_BN_IDX_l1b3,
                        layer1_shortcut_0_weight, P3_BN_IDX_l1sc);

    /* block2: 8→12, stride=(2,1) */
    Project3_BCResBlock(g_project3_act_b, g_project3_act_a,
                        8, 20, PROJECT3_MODEL_FRAMES, 12, 10, PROJECT3_MODEL_FRAMES, 2, 1,
                        layer2_conv1_weight, P3_BN_IDX_l2b1,
                        layer2_dwconv_weight, P3_BN_IDX_l2b2,
                        layer2_conv2_weight, P3_BN_IDX_l2b3,
                        layer2_shortcut_0_weight, P3_BN_IDX_l2sc);

    /* block3: 12→16, stride=(2,1) */
    Project3_BCResBlock(g_project3_act_a, g_project3_act_b,
                        12, 10, PROJECT3_MODEL_FRAMES, 16, 5, PROJECT3_MODEL_FRAMES, 2, 1,
                        layer3_conv1_weight, P3_BN_IDX_l3b1,
                        layer3_dwconv_weight, P3_BN_IDX_l3b2,
                        layer3_conv2_weight, P3_BN_IDX_l3b3,
                        layer3_shortcut_0_weight, P3_BN_IDX_l3sc);

    /* dwconv: 3×3 DW + BN + ReLU */
    Project3_DepthwiseConv2d(g_project3_act_b, g_project3_act_a,
                             16, 5, PROJECT3_MODEL_FRAMES, 5, PROJECT3_MODEL_FRAMES,
                             3, 3, 1, 1, 1, 1, dwconv_weight);
    Project3_ScaleBiasReLU(g_project3_act_a, 16, 5, PROJECT3_MODEL_FRAMES, P3_BN_IDX_dw, 1);

    /* pwconv: 16→20, 1×1 + BN + ReLU */
    Project3_Conv2d(g_project3_act_a, g_project3_act_b,
                    16, 5, PROJECT3_MODEL_FRAMES, 20, 5, PROJECT3_MODEL_FRAMES,
                    1, 1, 1, 1, 0, 0, pwconv_weight);
    Project3_ScaleBiasReLU(g_project3_act_b, 20, 5, PROJECT3_MODEL_FRAMES, P3_BN_IDX_pw, 1);

    /* conv2: 20→20, 5×1 + BN + ReLU */
    Project3_Conv2d(g_project3_act_b, g_project3_act_a,
                    20, 5, PROJECT3_MODEL_FRAMES, 20, 1, PROJECT3_MODEL_FRAMES,
                    5, 1, 1, 1, 0, 0, conv2_weight);
    Project3_ScaleBiasReLU(g_project3_act_a, 20, 1, PROJECT3_MODEL_FRAMES, P3_BN_IDX_conv2, 1);

    /* expand: 20→32, 1×1 + BN + ReLU */
    Project3_Conv2d(g_project3_act_a, g_project3_act_b,
                    20, 1, PROJECT3_MODEL_FRAMES, 32, 1, PROJECT3_MODEL_FRAMES,
                    1, 1, 1, 1, 0, 0, conv_expand_weight);
    Project3_ScaleBiasReLU(g_project3_act_b, 32, 1, PROJECT3_MODEL_FRAMES, P3_BN_IDX_expand, 1);

    /* Global AvgPool over time → [32] */
    {
        float inv_frames = 1.0f / (float)PROJECT3_MODEL_FRAMES;
        for (c = 0; c < 32; c++) {
            float sum = 0.0f;
            const float *restrict src = &g_project3_act_b[c * PROJECT3_MODEL_FRAMES];
            #pragma MUST_ITERATE(1, , 4)
            for (w = 0; w < PROJECT3_MODEL_FRAMES; w++) {
                sum += src[w];
            }
            g_project3_fc_in[c] = sum * inv_frames;
        }
    }

    /* FC: 32→13 */
    for (cls = 0; cls < PROJECT3_CMD_COUNT; cls++) {
        float sum = fc_bias[cls];
        const float *restrict wt = &fc_weight[cls * 32];
        #pragma MUST_ITERATE(1, , 4)
        for (i = 0; i < 32; i++) {
            sum += g_project3_fc_in[i] * wt[i];
        }
        logits[cls] = sum;
    }
}

static PROJECT3_INFER_RESULT Project3_LogitsToResult(const float *logits)
{
    PROJECT3_INFER_RESULT result;
    int i;
    int best = 0;
    float max_logit = logits[0];
    float sum_exp = 0.0f;
    float best_prob;

    for (i = 1; i < PROJECT3_CMD_COUNT; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            best = i;
        }
    }

    for (i = 0; i < PROJECT3_CMD_COUNT; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }

    best_prob = expf(logits[best] - max_logit) / sum_exp;
    result.class_id = (unsigned char)best;
    result.confidence = best_prob;
    result.valid = 1;

    if (best == PROJECT3_CLASS_SILENCE || best == PROJECT3_CLASS_UNKNOWN || best_prob < PROJECT3_INFERENCE_CONF_THRESHOLD) {
        result.valid = 0;
    }

    return result;
}

static unsigned char Project3_AcceptResult(PROJECT3_INFER_RESULT *result, const PROJECT3_UTTERANCE_BUFFER *utter)
{
    if (!result->valid) {
        result->class_id = PROJECT3_CLASS_UNKNOWN;
        return 0;
    }

    if (result->class_id == PROJECT3_CLASS_DOWN) {
        if (result->confidence < PROJECT3_DOWN_CONF_THRESHOLD || utter->count > PROJECT3_DOWN_MAX_SAMPLES) {
            result->class_id = PROJECT3_CLASS_UNKNOWN;
            result->valid = 0;
            return 0;
        }
    }

    return 1;
}

static void Project3_HandleSpeechEnd(PROJECT3_CONTEXT *ctx)
{
    PROJECT3_INFER_RESULT result;
    const char *label;

    Project3_SetAppState(ctx, PROJECT3_APP_SPEECH);

    result = Project3_RunInference(ctx, &ctx->utter);
    Project3_AcceptResult(&result, &ctx->utter);
    ctx->last_result = result;

    /* 静音：忽略，回到监听 */
    if (result.class_id == PROJECT3_CLASS_SILENCE) {
        ctx->input_gate_blocks = PROJECT3_POST_INFER_IGNORE_BLOCKS;
        Project3_ResetUtterance(ctx);
        ctx->last_main_text[0] = '\0';
        Project3_SetAppState(ctx, PROJECT3_APP_LISTENING);
        if (ctx->wake_active) {
            Project3_SetUiText(ctx, "Listening...", "Wake word: \"Zero\"", "Say Zero to deactivate");
        } else {
            Project3_SetUiText(ctx, "Say Zero...", "Wake word: \"Zero\"", "Speak Zero to activate");
        }
        return;
    }

    /* === 唤醒词 "zero" 处理 === */
    if (result.class_id == PROJECT3_CLASS_ZERO && result.confidence >= PROJECT3_CONFIDENCE_DISPLAY_THRESHOLD) {
        ctx->last_main_text[0] = '\0';
        if (ctx->wake_active) {
            /* 已激活 -> 听到zero，休眠 */
            ctx->wake_active = 0;
            Project3_SetAppState(ctx, PROJECT3_APP_LISTENING);
            Project3_SetUiText(ctx, "Sleeping...", "Wake word disabled", "Say Zero to activate");
            Led_Control(LED1_CORE, LED_OFF);
            Led_Control(LED2_CORE, LED_OFF);
        } else {
            /* 休眠 -> 听到zero，激活 */
            ctx->wake_active = 1;
            Project3_SetAppState(ctx, PROJECT3_APP_LISTENING);
            Project3_SetUiText(ctx, "Activated!", "Wake word enabled", "Speak any command");
            Led_Control(LED1_CORE, LED_ON);
            Led_Control(LED2_CORE, LED_ON);
        }
        ctx->input_gate_blocks = PROJECT3_POST_INFER_IGNORE_BLOCKS;
        ctx->ui_hold_blocks = PROJECT3_RESULT_HOLD_BLOCKS;
        Project3_ResetUtterance(ctx);
        return;
    }

    /* 未激活状态下，非zero唤醒词一律忽略 */
    if (!ctx->wake_active) {
        ctx->input_gate_blocks = PROJECT3_POST_INFER_IGNORE_BLOCKS;
        Project3_ResetUtterance(ctx);
        ctx->last_main_text[0] = '\0';
        Project3_SetAppState(ctx, PROJECT3_APP_LISTENING);
        Project3_SetUiText(ctx, "Say Zero...", "Wake word: \"Zero\"", "Speak Zero to activate");
        return;
    }

    /* === 激活状态下识别到其他词 === */
    label = Project3_GetLabel(result.class_id);

    /* 置信度低于50%%：提示重读 */
    if (result.confidence < PROJECT3_CONFIDENCE_DISPLAY_THRESHOLD) {
        ctx->last_main_text[0] = '\0';
        Project3_SetAppState(ctx, PROJECT3_APP_RESULT);
        Project3_SetUiText(ctx, "Please say again", "Confidence too low", "");
        ctx->ui_hold_blocks = PROJECT3_RESULT_HOLD_BLOCKS;
        ctx->input_gate_blocks = PROJECT3_POST_INFER_IGNORE_BLOCKS;
        Project3_ResetUtterance(ctx);
        return;
    }

    /* 置信度 >= 50%%，正常显示识别结果 */
    ctx->last_main_text[0] = '\0';
    Project3_SetAppState(ctx, PROJECT3_APP_RESULT);
    Project3_SetUiText(ctx, label, "", "");
    ctx->recognized_count++;
    ctx->ui_hold_blocks = PROJECT3_RESULT_HOLD_BLOCKS;

    // 点亮核心板LED作为识别成功反馈
    Led_Control(LED1_CORE, LED_ON);
    Led_Control(LED2_CORE, LED_ON);

    ctx->input_gate_blocks = PROJECT3_POST_INFER_IGNORE_BLOCKS;
    Project3_ResetUtterance(ctx);
}

static void Project3_RenderScreen(PROJECT3_CONTEXT *ctx, unsigned char force_redraw)
{
    unsigned long accent;
    int text_x;

    /* 防重入检查 */
    if (s_lcd_busy) {
        return;
    }

    /* 只在文字真正改变时重绘 */
    if (!force_redraw && strncmp(ctx->main_text, ctx->last_main_text, PROJECT3_RESULT_TEXT_LEN) == 0) {
        return;
    }

    s_lcd_busy = 1;

    /* 更新标题栏（状态灯颜色随状态变化） */
    Project3_DrawHeader(ctx);

    accent = Project3_GetAccentColor(ctx->app_state);

    /* 清除卡片内容区域 */
    Project3_ClearCardContent();

    /* 绘制主文字（48px 字体，居中） */
    GrContextForegroundSet(&Lcd_Context, accent);
    GrContextBackgroundSet(&Lcd_Context, ClrBlack);
    GrContextFontSet(&Lcd_Context, &g_sFontCm48);
    text_x = (P3_LCD_W - GrStringWidthGet(&Lcd_Context, ctx->main_text, -1)) / 2;
    if (text_x < P3_CARD_X + P3_CARD_MARGIN) text_x = P3_CARD_X + P3_CARD_MARGIN;
    GrStringDraw(&Lcd_Context, ctx->main_text, -1, text_x, P3_RESULT_TEXT_Y, 1);

    /* 结果状态下绘制置信度条 */
    Project3_DrawConfidenceBar(ctx);

    /* 清除卡片下方到底部状态栏之间的空白区域（旧widget残留） */
    {
        tRectangle gap;
        gap.sXMin = 0;
        gap.sYMin = P3_CARD_Y + P3_CARD_H;
        gap.sXMax = P3_LCD_W - 1;
        gap.sYMax = P3_BOTTOM_BAR_Y - 1;
        GrContextForegroundSet(&Lcd_Context, ClrBlack);
        GrRectFill(&Lcd_Context, &gap);
    }

    /* 动态更新底部状态栏文本 */
    {
        const char *state_str;
        switch (ctx->app_state) {
            case PROJECT3_APP_BOOT:        state_str = "BOOT";    break;
            case PROJECT3_APP_LISTENING:   state_str = "Listening"; break;
            case PROJECT3_APP_SPEECH:      state_str = "Recording"; break;
            case PROJECT3_APP_INFERENCING: state_str = "Thinking";  break;
            case PROJECT3_APP_RESULT:      state_str = "Result";    break;
            default:                       state_str = "Idle";      break;
        }
        snprintf(ctx->line1, PROJECT3_UI_TEXT_LEN, "%s | Recog: %lu | %s",
                 state_str, ctx->recognized_count,
                 ctx->wake_active ? "ACTIVE" : "SLEEP");
        snprintf(ctx->line2, PROJECT3_UI_TEXT_LEN, "VAD floor: %.1e | Energy: %.1e",
                 (double)ctx->vad.noise_floor, (double)ctx->vad.smooth_energy);
    }

    /* 底部状态栏：更新文本后重绘 */
    Project3_UpdateBottomStatusBar(ctx);

    /* 记录当前文字，避免重复绘制 */
    strncpy(ctx->last_main_text, ctx->main_text, PROJECT3_RESULT_TEXT_LEN);
    ctx->last_main_text[PROJECT3_RESULT_TEXT_LEN - 1] = '\0';

    s_lcd_busy = 0;
}

static void Project3_InitContext(PROJECT3_CONTEXT *ctx)
{
    memset(ctx, 0, sizeof(PROJECT3_CONTEXT));
    ctx->pass_through_enable = PROJECT3_PASS_THROUGH_ENABLE;
    ctx->model_state = PROJECT3_MODEL_READY;
    ctx->app_state = PROJECT3_APP_BOOT;
    ctx->vad.noise_floor = 1.0e-6f;
    ctx->vad.smooth_energy = 1.0e-6f;
    strcpy(ctx->main_text, "Booting...");
    strcpy(ctx->line1, "Wake word: \"Zero\"");
    strcpy(ctx->line2, "Say Zero to activate");
    ctx->redraw_needed = 1;
    Project3_InitTables();
}

static void Project3_InitUi(PROJECT3_CONTEXT *ctx)
{
    tRectangle fullscreen;

    Lcd_Init();

    /* 防重入标志初始化 */
    s_lcd_busy = 0;

    /* 全屏清屏 */
    fullscreen.sXMin = 0;
    fullscreen.sYMin = 0;
    fullscreen.sXMax = P3_LCD_W - 1;
    fullscreen.sYMax = P3_LCD_H - 1;
    GrContextForegroundSet(&Lcd_Context, ClrBlack);
    GrRectFill(&Lcd_Context, &fullscreen);

    /* 初始化 last_main_text 为空，确保第一次一定绘制 */
    ctx->last_main_text[0] = '\0';

    /* 等待LCD稳定 */
    volatile int i;
    for (i = 0; i < 100000; i++);

    /* 绘制静态布局元素 */
    Project3_DrawCardFrame();

    /* 绘制初始动态内容 */
    Project3_RenderScreen(ctx, 1);
}

static void Project3_HandleKeys(PROJECT3_CONTEXT *ctx)
{
    if (FLAG_KEY1) {
        FLAG_KEY1 = 0;
        /* KEY1: 手动切换唤醒状态（等同于说 zero 的唤醒/休眠效果） */
        Project3_ResetUtterance(ctx);
        ctx->input_gate_blocks = 0;
        ctx->ui_hold_blocks = PROJECT3_RESULT_HOLD_BLOCKS;
        if (ctx->wake_active) {
            ctx->wake_active = 0;
            Project3_SetAppState(ctx, PROJECT3_APP_LISTENING);
            Project3_SetUiText(ctx, "Sleeping...", "Wake word disabled", "Say Zero to activate");
            Led_Control(LED1_CORE, LED_OFF);
            Led_Control(LED2_CORE, LED_OFF);
        } else {
            ctx->wake_active = 1;
            Project3_SetAppState(ctx, PROJECT3_APP_LISTENING);
            Project3_SetUiText(ctx, "Activated!", "Wake word enabled", "Speak any command");
            Led_Control(LED1_CORE, LED_ON);
            Led_Control(LED2_CORE, LED_ON);
        }
    }
    if (FLAG_KEY2) {
        FLAG_KEY2 = 0;
        // KEY2: 降低VAD灵敏度（减少误触发）
        ctx->vad.noise_floor *= 1.15f;
    }
    if (FLAG_KEY3) {
        FLAG_KEY3 = 0;
        // KEY3: 提高VAD灵敏度（更容易触发）
        ctx->vad.noise_floor *= 0.85f;
    }
    if (FLAG_KEY4) {
        FLAG_KEY4 = 0;
        // KEY4: 重置VAD基线
        ctx->vad.noise_floor = 1.0e-6f;
        ctx->vad.smooth_energy = 1.0e-6f;
    }
    if (FLAG_KEY5) {
        FLAG_KEY5 = 0;
        // KEY5: 系统重启
        Project3_InitContext(ctx);
        Project3_ModelInit(ctx);
    }
}

static void Project3_HandleTouch(PROJECT3_CONTEXT *ctx)
{
    if (FLAG_TOUCH) {
        FLAG_TOUCH = 0;
        Touch_Scan();
    }

    // 触摸屏功能已禁用，所有控制通过物理按键
    if (FLAG_BUTTON_1) {
        FLAG_BUTTON_1 = 0;
    }
    if (FLAG_BUTTON_2) {
        FLAG_BUTTON_2 = 0;
    }
    if (FLAG_BUTTON_3) {
        FLAG_BUTTON_3 = 0;
    }
    if (FLAG_BUTTON_4) {
        FLAG_BUTTON_4 = 0;
    }
    if (FLAG_BUTTON_5) {
        FLAG_BUTTON_5 = 0;
    }
    if (FLAG_BUTTON_6) {
        FLAG_BUTTON_6 = 0;
    }
    if (FLAG_BUTTON_7) {
        FLAG_BUTTON_7 = 0;
    }
    if (FLAG_BUTTON_8) {
        FLAG_BUTTON_8 = 0;
    }
}

static void Project3_ProcessAudioBlock(PROJECT3_CONTEXT *ctx, short *block, unsigned int block_samples)
{
    unsigned int offset;
    unsigned char block_has_speech = 0;
    unsigned char end_pending = 0;

    Project3_ServiceUiHold(ctx);

    if (ctx->input_gate_blocks > 0) {
        ctx->input_gate_blocks--;
        return;
    }

    if (ctx->ui_hold_blocks > 0) {
        return;
    }

    for (offset = 0; offset + PROJECT3_VAD_FRAME_LEN <= block_samples; offset += PROJECT3_VAD_HOP) {
        float frame_energy = Project3_ComputeFrameEnergy(&block[offset], PROJECT3_VAD_FRAME_LEN);
        unsigned char started = Project3_UpdateVad(ctx, frame_energy);

        ctx->frame_counter++;

        if (started) {
            ctx->utter.ready = 0;
        }

        if (ctx->vad.active || started) {
            block_has_speech = 1;
        }

        if (Project3_ShouldEndUtterance(ctx)) {
            end_pending = 1;
            block_has_speech = 1;
            break;
        }
    }

    if (block_has_speech) {
        Project3_AppendRawBlock(ctx, block, block_samples);
    }

    if (end_pending || ctx->utter.count >= PROJECT3_RAW_MAX_SAMPLES) {
        if (ctx->utter.count >= PROJECT3_MIN_UTTERANCE_SAMPLES) {
            Project3_HandleSpeechEnd(ctx);
        } else {
            Project3_ResetUtterance(ctx);
            ctx->input_gate_blocks = 0;
        }
    }
}

static void Project3_ModelInit(PROJECT3_CONTEXT *ctx)
{
    Project3_InitMergedBN();  /* 预计算合并 BN 参数, 消除推理时 sqrt/div */
    ctx->model_state = PROJECT3_MODEL_READY;
    ctx->wake_active = 0;
    Project3_SetAppState(ctx, PROJECT3_APP_LISTENING);
    Project3_SetUiText(ctx, "Say Zero...", "Wake word: \"Zero\"", "Speak Zero to activate");
}

static PROJECT3_INFER_RESULT Project3_RunInference(PROJECT3_CONTEXT *ctx, const PROJECT3_UTTERANCE_BUFFER *utter)
{
    ctx->model_state = PROJECT3_MODEL_READY;
    Project3_ExtractLogMel(utter, g_project3_logmel);
    Project3_BCResNetForward(g_project3_logmel, g_project3_logits);
    return Project3_LogitsToResult(g_project3_logits);
}

static const char *Project3_GetLabel(unsigned char class_id)
{
    if (class_id >= PROJECT3_CMD_COUNT) {
        return "_unknown_";
    }
    return g_project3_labels[class_id];
}

static void Project3_ResetUtterance(PROJECT3_CONTEXT *ctx)
{
    ctx->utter.count = 0;
    ctx->utter.ready = 0;
    ctx->vad.active = 0;
    ctx->vad.speech_hold_frames = 0;
    ctx->vad.silence_hold_frames = 0;
}

static void Project3_ServiceUiHold(PROJECT3_CONTEXT *ctx)
{
    if (ctx->ui_hold_blocks > 0) {
        ctx->ui_hold_blocks--;
        if (ctx->ui_hold_blocks == 0) {
            // 强制标记 last_main_text 为空，确保下次一定重绘
            ctx->last_main_text[0] = '\0';
            Project3_SetAppState(ctx, PROJECT3_APP_LISTENING);
            if (ctx->wake_active) {
                Project3_SetUiText(ctx, "Listening...", "Wake word: \"Zero\"", "Say Zero to deactivate");
                Led_Control(LED1_CORE, LED_ON);
                Led_Control(LED2_CORE, LED_ON);
            } else {
                Project3_SetUiText(ctx, "Say Zero...", "Wake word: \"Zero\"", "Speak Zero to activate");
                Led_Control(LED1_CORE, LED_OFF);
                Led_Control(LED2_CORE, LED_OFF);
            }
        }
    }
}

static void Project3_UpdateUi(PROJECT3_CONTEXT *ctx, unsigned char force_redraw)
{
    if (force_redraw || ctx->redraw_needed) {
        Project3_RenderScreen(ctx, 1);
        ctx->redraw_needed = 0;
    }
}


#endif
