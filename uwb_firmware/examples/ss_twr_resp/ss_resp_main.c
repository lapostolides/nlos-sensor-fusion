/*! ----------------------------------------------------------------------------
 * @file    ss_resp_main.c
 * @brief   CIR Reader - Increment 2 (binary UART output)
 *
 *          Listens for any UWB frame, reads the DW1000 CIR accumulator,
 *          and emits a binary frame over UART at 921600 baud.
 *
 *          Binary frame format (4073 bytes):
 *            [0xBC][0xAD]   2B  sync
 *            [seq]          2B  uint16 LE frame counter
 *            [fp_index]     2B  uint16 LE first path index
 *            [rxpacc]       2B  uint16 LE preamble accumulation count
 *            [cir]       4064B  1016 x (I: int16 LE, Q: int16 LE)
 *            [xor]          1B  XOR checksum of bytes[2:-1]
 * --------------------------------------------------------------------------*/

#include "sdk_config.h"
#include <string.h>
#include "FreeRTOS.h"
#include "task.h"
#include "deca_device_api.h"
#include "deca_regs.h"
#include "port_platform.h"
#include "nrf.h"

/* -- Raw polled UART (bypasses broken app_uart driver) -------------------- */

static void uart_putc(uint8 c)
{
    NRF_UART0->EVENTS_TXDRDY = 0;
    NRF_UART0->TXD = c;
    while (NRF_UART0->EVENTS_TXDRDY == 0) {}
}

static void uart_send_str(const char *s)
{
    while (*s)
        uart_putc((uint8)*s++);
}

/* -- CIR buffer ----------------------------------------------------------- */

#define CIR_N_SAMPLES  1016u
#define CIR_BUF_BYTES  (CIR_N_SAMPLES * 4u)

/* +1 for the dummy byte that dwt_readaccdata always prepends */
static uint8 cir_buf[CIR_BUF_BYTES + 1];

/* -- Binary frame transmit ------------------------------------------------ */

static void send_cir_frame(uint16 seq, uint16 fp_index, uint16 rxpacc)
{
    uint32 i;
    uint8  xor_val = 0;
    uint8  b;

#define SEND_XOR(byte) \
    do { b = (uint8)(byte); uart_putc(b); xor_val ^= b; } while (0)

    uart_putc(0xBC);
    uart_putc(0xAD);

    SEND_XOR(seq       & 0xFF);  SEND_XOR(seq       >> 8);
    SEND_XOR(fp_index  & 0xFF);  SEND_XOR(fp_index  >> 8);
    SEND_XOR(rxpacc    & 0xFF);  SEND_XOR(rxpacc    >> 8);

    for (i = 0; i < CIR_BUF_BYTES; i++)
        SEND_XOR(cir_buf[i + 1]);

    uart_putc(xor_val);

#undef SEND_XOR
}

void ss_responder_task_function(void *pvParameter)
{
    uint16 seq        = 0;
    uint32 status_reg = 0;
    uint32 heartbeat  = 0;

    UNUSED_PARAMETER(pvParameter);

    dwt_setleds(DWT_LEDS_ENABLE);

    vTaskDelay(100);
    uart_send_str("CIR:RDY\r\n");

    while (1)
    {
        /* Periodic heartbeat so PuTTY can catch it at any time */
        if (++heartbeat % 200 == 0)
            uart_send_str("HB\r\n");

        dwt_rxenable(DWT_START_RX_IMMEDIATE);

        while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) &
                 (SYS_STATUS_RXFCG | SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR)))
        {}

        if (status_reg & SYS_STATUS_RXFCG)
        {
            uint16 fp_index, rxpacc;

            dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG);

            fp_index = dwt_read16bitoffsetreg(RX_TIME_ID, RX_TIME_FP_INDEX_OFFSET);
            rxpacc   = (dwt_read32bitreg(RX_FINFO_ID) >> RX_FINFO_RXPACC_SHIFT) & 0x0FFF;

            dwt_readaccdata(cir_buf, CIR_BUF_BYTES + 1, 0);
            send_cir_frame(seq, fp_index, rxpacc);

            LEDS_INVERT(BSP_LED_0_MASK);
            seq++;
        }
        else
        {
            dwt_write32bitreg(SYS_STATUS_ID,
                              SYS_STATUS_ALL_RX_ERR | SYS_STATUS_ALL_RX_TO);
            dwt_rxreset();
        }

        vTaskDelay(1);
    }
}
