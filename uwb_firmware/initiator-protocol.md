# ss_twr_init_ts — Initiator Protocol

TX-timestamp initiator firmware for the DWM1001-DEV. Sends a UWB poll frame at a fixed interval and emits the DW1000 TX timestamp over UART as a compact binary frame. No response is expected from the companion responder (`ss_twr_resp`), which captures CIR instead of sending a ranging reply.

## Source Files

| File | Purpose |
|---|---|
| `main.c` | Hardware init: UART, DW1000 reset/config, FreeRTOS task creation |
| `ss_init_main.c` | Initiator task: TX loop, timestamp readout, binary frame TX |
| `UART/UART.h` | UART init/getc interface (not used in the hot path) |
| `config/sdk_config.h` | nRF5 SDK configuration |

---

## DW1000 Radio Configuration

Defined in `main.c` as `dwt_config_t config`. Must match the responder's config exactly:

| Parameter | Value | Notes |
|---|---|---|
| Channel | 5 | UWB channel 5 (~6.5 GHz centre) |
| PRF | 64 MHz | Pulse repetition frequency |
| Preamble length | 256 symbols | `DWT_PLEN_256` |
| PAC size | 16 | `DWT_PAC16` — recommended for preamble 256 |
| TX preamble code | 10 | Must match responder |
| RX preamble code | 10 | Must match responder |
| SFD type | Standard (0) | IEEE 802.15.4a standard SFD |
| Data rate | 6.8 Mbps | `DWT_BR_6M8` |
| PHY header mode | Standard | `DWT_PHRMODE_STD` |
| SFD timeout | 249 symbols | `256 + 1 + 8 - 16` (RX-only, not relevant here) |

### Antenna Delays

Defined locally in `main.c` (override `port_platform.h` defaults):

| Constant | Value (DWT ticks) | Notes |
|---|---|---|
| `TX_ANT_DLY` | **16300** | Applied to TX antenna — differs from responder |
| `RX_ANT_DLY` | 16456 | Applied to RX antenna |

Note: The initiator TX delay (16300) differs from the responder's (16456). Both should be calibrated per-device. The TX timestamp reported over UART already incorporates `TX_ANT_DLY` via the DW1000 hardware.

1 DWT tick ≈ 15.65 ps (DW1000 internal 64 GHz clock).

### Legacy RX Settings

These are set in `main.c` but have no effect — RX is never enabled in the TX-only initiator task:

| Setting | Value | Notes |
|---|---|---|
| `dwt_setrxaftertxdelay` | 100 UWB µs | Leftover from original SS-TWR; unused |
| `dwt_setrxtimeout` | 65000 (~65 ms) | Leftover from original SS-TWR; unused |
| Preamble detect timeout | disabled | `dwt_setpreambledetecttimeout` call commented out |

---

## UART Configuration

Configured by direct register writes in `main.c` (bypasses the broken `app_uart` nRF SDK driver):

| Register | Value | Meaning |
|---|---|---|
| `NRF_UART0->PSELTXD` | 5 | TX on GPIO P0.05 (DWM1001-DEV J-Link UART TX) |
| `NRF_UART0->PSELRXD` | 0xFFFFFFFF | RX disconnected (TX-only) |
| `NRF_UART0->BAUDRATE` | `0x01D7E000` | **115200 baud** |
| `NRF_UART0->CONFIG` | 0 | No parity, no flow control |

Data is sent byte-by-byte via polled writes in `uart_putc()`. No DMA or interrupt-driven TX.

---

## Poll Frame

The UWB payload transmitted each iteration:

```c
static uint8 tx_poll_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'W', 'A', 'V', 'E', 0xE0, 0, 0};
```

Total: **12 bytes** (including 2-byte CRC appended by DW1000 hardware).

| Offset | Value | Description |
|---|---|---|
| 0–1 | `0x41, 0x88` | IEEE 802.15.4 frame control (data frame, short addr, PAN compression) |
| 2 | `frame_seq_nb` | Sequence number — overwritten each iteration |
| 3–4 | `0xCA, 0xDE` | Destination PAN ID |
| 5–8 | `'W','A','V','E'` | Destination address |
| 9 | `0xE0` | Function code (poll) |
| 10–11 | `0x00, 0x00` | CRC placeholder — filled by DW1000 hardware |

The ranging bit is set in the PHY header via `dwt_writetxfctrl(sizeof(tx_poll_msg), 0, 1)` (third argument = ranging enable).

---

## TX Loop Behavior

```
write poll frame data → queue for TX (dwt_writetxdata / dwt_writetxfctrl)
  │
  └─ dwt_starttx(DWT_START_TX_IMMEDIATE)
        │
        └─ poll SYS_STATUS until TXFRS set
              │
              └─ clear TXFRS → dwt_readtxtimestamp() → send_tx_ts_frame()
                    │
                    └─ toggle LED0, increment frame_seq_nb
                          │
                          └─ vTaskDelay(RNG_DELAY_MS)  // 100 ms
```

Transmit rate: **~10 Hz** (100 ms delay per iteration, plus TX air time of ~0.4 ms at 6.8 Mbps with 256-symbol preamble).

### Sequence Number

```c
static uint8 frame_seq_nb = 0;   // 8-bit, wraps at 256
```

`frame_seq_nb` is `uint8` — it wraps every 256 frames. It is cast to `uint16` when building the binary UART frame, so the UART frame `seq` field also wraps at 256 (values 0–255 only, upper byte always 0).

The SN embedded in the UWB poll frame (`tx_poll_msg[2]`) is the same `uint8` value.

---

## Binary Frame Format

Total size: **10 bytes**

```
Offset  Size  Type         Field
------  ----  -----------  -----
0       2     uint8[2]     Sync bytes: 0xBC, 0xAD  (not included in XOR)
2       2     uint16 LE    seq    — frame counter (uint8 cast to uint16; wraps at 256)
4       5     uint8[5] LE  tx_ts  — 40-bit DW1000 TX timestamp
9       1     uint8        xor    — XOR of bytes[2..8] inclusive
```

### XOR Checksum

Covers `seq` and `tx_ts` (bytes[2:-1]). Sync bytes excluded. Same scheme as the responder's CIR frame.

### TX Timestamp

Read via `dwt_readtxtimestamp()` into a 5-byte array after `TXFRS` is set.

- DW1000 40-bit TX timestamp, little-endian
- Resolution: ~15.65 ps/tick
- Represents the actual over-the-air TX time, adjusted by `TX_ANT_DLY`

---

## FreeRTOS Task

Defined and created in `main.c`, implemented in `ss_init_main.c`:

```c
xTaskCreate(ss_initiator_task_function, "SSTWR_INIT",
            configMINIMAL_STACK_SIZE + 200, NULL, 2, &ss_initiator_task_handle);
```

| Parameter | Value |
|---|---|
| Task name | `"SSTWR_INIT"` |
| Stack size | `configMINIMAL_STACK_SIZE + 200` words (~1000 bytes, platform-dependent) |
| Priority | 2 |
| Delay per iteration | `RNG_DELAY_MS` = 100 ticks (100 ms) |

Stack is significantly smaller than the responder's because no CIR buffer is allocated on the stack or as a static global — only a 5-byte `tx_ts` array lives on the task stack.

A LED toggle task (`"LED0"`, `configMINIMAL_STACK_SIZE + 200`, priority 2) blinks every 200 ms, and a FreeRTOS timer toggles LED1 every 2000 ms.

---

## RAM Summary

| Symbol | Location | Size |
|---|---|---|
| `tx_poll_msg[]` | BSS (static global) | 12 bytes |
| `frame_seq_nb` | BSS (static global) | 1 byte |
| `tx_ts[5]` | Task stack (local) | 5 bytes |
| `ss_initiator_task_function` stack | FreeRTOS heap | ~1000 bytes |
| `led_toggle_task_function` stack | FreeRTOS heap | ~1000 bytes |

No large static buffers (contrast with responder's 4065-byte `cir_buf`).

---

## Host Capture

[`capture_uwb.py`](../../../capture_uwb.py) parses the binary stream:

```python
SYNC        = bytes([0xBC, 0xAD])
INIT_FRAME  = 10    # total bytes per frame
```

Output arrays saved to `init.npz`:

| Array | Shape | dtype | Description |
|---|---|---|---|
| `seq` | (N,) | uint16 | Frame counter (wraps at 256) |
| `tx_ts` | (N,) | uint64 | 40-bit TX timestamp |
| `timestamp` | (N,) | float64 | Host `time.monotonic()`, zeroed to first frame |

---

## Pairing with the Responder

The initiator and responder share the same sync bytes (`0xBC 0xAD`) and XOR checksum scheme, but their frames differ in size (10 vs 4078 bytes). `capture_uwb.py` dispatches to the correct parser based on which port each board is connected to.

To correlate a TX timestamp (initiator) with a CIR capture (responder), match on `seq` — both counters increment once per UWB poll frame and start at 0 on boot. Because `frame_seq_nb` wraps at 256 and the responder's `seq` is `uint16` (wraps at 65536), long captures will show the initiator `seq` cycling while the responder's continues counting. Account for this with modular arithmetic when aligning.
