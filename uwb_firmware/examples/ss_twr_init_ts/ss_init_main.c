/*! ----------------------------------------------------------------------------
 * @file    ss_init_main.c
 * @brief   CIR Initiator - TX timestamp UART output
 *
 *          Sends a UWB poll frame every RNG_DELAY_MS and emits the DW1000
 *          TX timestamp over UART as a compact binary frame.  No response is
 *          expected from the companion responder (which sends CIR data instead
 *          of a ranging reply).
 *
 *          Binary frame format (10 bytes):
 *            [0xBC][0xAD]   2B  sync
 *            [seq]          2B  uint16 LE frame counter (matches poll SN)
 *            [tx_ts]        5B  uint40 LE DW1000 TX timestamp (~15.65 ps/tick)
 *            [xor]          1B  XOR checksum of bytes[2:-1]
 * --------------------------------------------------------------------------*/

#include <string.h>
#include "FreeRTOS.h"
#include "task.h"
#include "deca_device_api.h"
#include "deca_regs.h"
#include "port_platform.h"
#include "nrf.h"

/* Inter-ranging delay period, in milliseconds. */
#define RNG_DELAY_MS 100

/* Poll frame sent to trigger ranging on the responder. */
static uint8 tx_poll_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'W', 'A', 'V', 'E', 0xE0, 0, 0};
#define ALL_MSG_SN_IDX 2

/* Frame sequence number, incremented after each transmission. */
static uint8 frame_seq_nb = 0;

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

/* -- Binary frame transmit ------------------------------------------------ */

static void send_tx_ts_frame(uint16 seq, const uint8 *tx_ts)
{
    uint8  i;
    uint8  xor_val = 0;
    uint8  b;

#define SEND_XOR(byte) \
    do { b = (uint8)(byte); uart_putc(b); xor_val ^= b; } while (0)

    uart_putc(0xBC);
    uart_putc(0xAD);

    SEND_XOR(seq & 0xFF);  SEND_XOR(seq >> 8);

    /* DW1000 40-bit TX timestamp, 5 bytes little-endian */
    for (i = 0; i < 5; i++)
        SEND_XOR(tx_ts[i]);

    uart_putc(xor_val);

#undef SEND_XOR
}

/**@brief SS TWR Initiator task entry function.
 *
 * Sends poll frames and emits the DW1000 TX timestamp over UART after each
 * transmission.  The companion responder does not send a ranging reply, so
 * no response handling is needed here.
 *
 * @param[in] pvParameter   Pointer that will be used as the parameter for the task.
 */
void ss_initiator_task_function(void *pvParameter)
{
    UNUSED_PARAMETER(pvParameter);

    dwt_setleds(DWT_LEDS_ENABLE);

    vTaskDelay(100);
    uart_send_str("TX-TS:RDY\r\n");

    while (1)
    {
        uint8  tx_ts[5];
        uint32 status_reg;

        /* Write poll frame data and queue for transmission. */
        tx_poll_msg[ALL_MSG_SN_IDX] = frame_seq_nb;
        dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS);
        dwt_writetxdata(sizeof(tx_poll_msg), tx_poll_msg, 0);
        dwt_writetxfctrl(sizeof(tx_poll_msg), 0, 1);

        /* Transmit without enabling RX — responder does not send a reply. */
        dwt_starttx(DWT_START_TX_IMMEDIATE);

        /* Wait for TX frame sent (TXFRS). */
        while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & SYS_STATUS_TXFRS))
        {}
        dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS);

        /* Read TX timestamp — valid now that TXFRS is set. */
        dwt_readtxtimestamp(tx_ts);
        send_tx_ts_frame((uint16)frame_seq_nb, tx_ts);

        LEDS_INVERT(BSP_LED_0_MASK);
        frame_seq_nb++;

        vTaskDelay(RNG_DELAY_MS);
    }
}
