#include "project3.h"

int main(void)
{
    static PROJECT3_CONTEXT ctx;
    unsigned int i;

    Project3_InitContext(&ctx);

    Sys_Init();
    Key_Init();
    Touch_Init();
    Project3_InitUi(&ctx);

    Adc_Init(PROJECT3_ADC_RATE, PROJECT3_BLOCK_SAMPLES);
    Dac_Init(PROJECT3_DAC_RATE, PROJECT3_BLOCK_SAMPLES, PROJECT3_DAC_CHANNEL_MASK);

    for (i = 0; i < PROJECT3_BLOCK_SAMPLES; i++) {
        DA_CH1_Buf0[i] = 0;
        DA_CH1_Buf1[i] = 0;
    }

    Project3_ModelInit(&ctx);
    Project3_UpdateUi(&ctx, 1);

    Adc_Start();
    Dac_Start();

    while (1) {
        // LCD刷新独立于音频处理，在主循环空闲时执行
        if (ctx.redraw_needed && !s_lcd_busy && FLAG_AD == 0) {
            Project3_UpdateUi(&ctx, 0);
        }

        Project3_HandleKeys(&ctx);
        Project3_HandleTouch(&ctx);

        if (FLAG_AD == 1) {
            FLAG_AD = 0;
            Project3_CopyInputBlock(g_project3_input_block);
            Project3_ProcessAudioBlock(&ctx, g_project3_input_block, PROJECT3_BLOCK_SAMPLES);
            Project3_FillDacOutput(&ctx, g_project3_input_block);
            Project3_WriteOutputBlockToDac(g_project3_output_block);
            FLAG_DA = 0;
        }
    }
}
