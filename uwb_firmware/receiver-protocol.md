# ss_twr_resp — Receiver Protocol

CIR (Channel Impulse Response) receiver firmware for the DWM1001-DEV. Listens for any incoming UWB frame, reads the full DW1000 CIR accumulator, and emits a compact binary frame over UART. No ranging reply is sent.

## Source Files

| File | Purpose |
|---|---|
| `main.c` | Hardware init: UART, DW1000 reset/config, FreeRTOS task creation |
| `ss_resp_main.c` | Receiver task: RX loop, CIR readout, binary frame TX |
| `UART/UART.h` | UART init/getc interface (not used in the hot path; direct register writes are used instead) |
| `config/sdk_config.h` | nRF5 SDK configuration |

---

## DW1000 Radio Configuration

Defined in `main.c` as `dwt_config_t config`:

| Parameter | Value | Notes |
|---|---|---|
| Channel | 5 | UWB channel 5 (~6.5 GHz centre) |
| PRF | 64 MHz | Pulse repetition frequency |
| Preamble length | 256 symbols | `DWT_PLEN_256` |
| PAC size | 16 | `DWT_PAC16` — recommended for preamble 256 |
| TX preamble code | 10 | Must match initiator |
| RX preamble code | 10 | Must match initiator |
| SFD type | Standard (0) | IEEE 802.15.4a standard SFD |
| Data rate | 6.8 Mbps | `DWT_BR_6M8` |
| PHY header mode | Standard | `DWT_PHRMODE_STD` |
| SFD timeout | 249 symbols | `256 + 1 + 8 - 16` (preamble + 1 + SFD_len - PAC) |
| RX timeout | 0 (disabled) | Wait indefinitely for a frame |
| Preamble detect timeout | 1000 PAC periods | Set via `dwt_setpreambledetecttimeout(1000)` |

### Antenna Delays

Defined in `deca_driver/port/port_platform.h`:

| Constant | Value (DWT ticks) | Notes |
|---|---|---|
| `TX_ANT_DLY` | 16456 | Applied to TX antenna |
| `RX_ANT_DLY` | 16456 | Applied to RX antenna |

Both set to the same default value. Sum represents total TX→RX antenna delay. Should be calibrated per-device for accurate ranging; a small positive distance error is expected with these defaults.

1 DWT tick ≈ 15.65 ps (DW1000 internal 64 GHz clock).

---

## UART Configuration

Configured by direct register writes in `main.c` (bypasses the broken `app_uart` nRF SDK driver):

| Register | Value | Meaning |
|---|---|---|
| `NRF_UART0->PSELTXD` | 5 | TX on GPIO P0.05 (DWM1001-DEV J-Link UART TX) |
| `NRF_UART0->PSELRXD` | 0xFFFFFFFF | RX disconnected (TX-only) |
| `NRF_UART0->BAUDRATE` | `0x01D7E000` | **115200 baud** |
| `NRF_UART0->CONFIG` | 0 | No parity, no flow control |

Note: The file header comment in `ss_resp_main.c` says "921600 baud" — this is a stale comment. The actual baud rate is **115200**, matching the `capture_uwb.py` default.

Data is sent byte-by-byte via polled writes in `uart_putc()`. There is no DMA or interrupt-driven TX.

---

## CIR Accumulator

The DW1000 CIR accumulator contains the complex baseband channel response sampled at the chip rate.

| Parameter | Value |
|---|---|
| Number of samples | 1016 |
| Sample format | Complex: I (int16 LE) + Q (int16 LE) |
| Bytes per sample | 4 |
| Total CIR payload | 4064 bytes |

### RAM Buffer

```c
#define CIR_N_SAMPLES  1016u
#define CIR_BUF_BYTES  (CIR_N_SAMPLES * 4u)   // 4064

/* +1 for the dummy byte dwt_readaccdata() always prepends */
static uint8 cir_buf[CIR_BUF_BYTES + 1];       // 4065 bytes, static global
```

`dwt_readaccdata()` always writes a dummy byte at index 0. The actual IQ data starts at `cir_buf[1]`. The frame transmit loop skips this dummy byte when sending.

### Read Order

The accumulator **must** be read before clearing the RX status register or re-enabling the receiver, otherwise the buffer is overwritten. The firmware enforces this order:

1. `dwt_readaccdata()` — read CIR
2. `dwt_read16bitoffsetreg(RX_TIME_ID, ...)` — read first path index
3. `dwt_read32bitreg(RX_FINFO_ID)` — read preamble accumulation count
4. `dwt_readrxtimestamp()` — read RX timestamp
5. `dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG)` — clear RX good flag
6. `send_cir_frame()` — transmit binary frame over UART

---

## Metadata Fields

### First Path Index (`fp_index`)

Read from `RX_TIME_ID` register at offset `RX_TIME_FP_INDEX_OFFSET`.

- DW1000 reports this as a **10.6 fixed-point** value (16-bit: upper 10 bits = integer bin, lower 6 bits = fractional)
- Transmitted in the binary frame as the raw 16-bit register value
- `capture_uwb.py` converts to integer bin: `fp_raw >> 6`, clamped to `[0, 1015]`
- Indicates which CIR sample corresponds to the first detected path (useful for ToA estimation)

### Preamble Accumulation Count (`rxpacc`)

Read from `RX_FINFO_ID` register, bits [19:8] (`>> RX_FINFO_RXPACC_SHIFT`, masked `& 0x0FFF`).

- 12-bit value (0–4095)
- Number of preamble symbols accumulated; used to normalize CIR amplitude
- Higher value → better SNR estimate

### RX Timestamp (`rx_ts`)

Read via `dwt_readrxtimestamp()` into a 5-byte array.

- DW1000 40-bit RX timestamp, little-endian
- Resolution: ~15.65 ps/tick
- Represents the time of arrival of the first path, adjusted by the programmed `RX_ANT_DLY`

---

## Binary Frame Format

Total size: **4078 bytes**

```
Offset  Size  Type         Field
------  ----  -----------  -----
0       2     uint8[2]     Sync bytes: 0xBC, 0xAD  (not included in XOR)
2       2     uint16 LE    seq       — frame counter, increments each valid RX
4       2     uint16 LE    fp_index  — first path index (10.6 fixed point, raw)
6       2     uint16 LE    rxpacc    — preamble accumulation count
8       5     uint8[5] LE  rx_ts     — 40-bit DW1000 RX timestamp
13   4064     int16 LE[]   cir       — 1016 complex samples (I, Q interleaved)
4077    1     uint8        xor       — XOR of bytes[2..4076] inclusive
```

### XOR Checksum

Covers all bytes from `seq` through the last CIR byte (i.e., `bytes[2:-1]`). The sync bytes (0xBC, 0xAD) are excluded. Computed and verified in `capture_uwb.py` before decoding CIR data.

### CIR Layout

Each sample is two consecutive `int16 LE` values: `[I, Q]`. The array is 1016 pairs = 2032 int16 values = 4064 bytes.

```python
iq = np.frombuffer(payload[13:], dtype="<i2").reshape(1016, 2)
cir = iq[:, 0] + 1j * iq[:, 1]   # complex64
```

---

## FreeRTOS Task

Defined in `main.c`, implemented in `ss_resp_main.c`:

```c
xTaskCreate(ss_responder_task_function, "SSTWR_RESP", 2500, NULL, 2, &ss_responder_task_handle);
```

| Parameter | Value |
|---|---|
| Task name | `"SSTWR_RESP"` |
| Stack size | 2500 words = **10000 bytes** |
| Priority | 2 |
| Delay per iteration | 1 tick (`vTaskDelay(1)`) |

A second LED toggle task also runs (`"LED0"`, stack `configMINIMAL_STACK_SIZE + 200`, priority 2) and a FreeRTOS timer toggles LED1 every 2000 ms.

---

## Receiver Loop Behavior

```
dwt_rxenable(DWT_START_RX_IMMEDIATE)
  │
  └─ poll SYS_STATUS until RXFCG | RX_TIMEOUT | RX_ERROR
        │
        ├─ RXFCG (good frame):
        │     read CIR → read fp_index, rxpacc, rx_ts → clear RXFCG → send frame
        │     toggle LED0, increment seq
        │
        └─ error/timeout:
              clear error flags, call dwt_rxreset(), re-enable RX
```

On error, the frame is discarded and `seq` is not incremented. A gap in `seq` values on the host indicates dropped frames (missed receptions or CRC errors).

---

## RAM Summary

| Symbol | Location | Size |
|---|---|---|
| `cir_buf[]` | BSS (static global) | 4065 bytes |
| `ss_responder_task_function` stack | FreeRTOS heap | 10000 bytes |
| `led_toggle_task_function` stack | FreeRTOS heap | ~1000 bytes (configMINIMAL_STACK_SIZE + 200 words) |
| DW1000 driver state | BSS | ~varies (dwt internal) |

---

## Host Capture

[`capture_uwb.py`](../../../capture_uwb.py) parses the binary stream:

```python
SYNC       = bytes([0xBC, 0xAD])
RESP_FRAME = 4078   # total bytes per frame
```

Output arrays saved to `resp.npz`:

| Array | Shape | dtype | Description |
|---|---|---|---|
| `cir` | (N, 1016) | complex64 | I + jQ |
| `cir_i` | (N, 1016) | float32 | In-phase |
| `cir_q` | (N, 1016) | float32 | Quadrature |
| `cir_mag` | (N, 1016) | float32 | `\|I + jQ\|` |
| `cir_phase` | (N, 1016) | float32 | `angle(I + jQ)` radians |
| `seq` | (N,) | uint16 | Frame counter |
| `fp_index` | (N,) | uint16 | First-path bin (integer, after `>> 6`) |
| `rxpacc` | (N,) | uint16 | Preamble accumulation count |
| `rx_ts` | (N,) | uint64 | 40-bit RX timestamp |
| `timestamp` | (N,) | float64 | Host `time.monotonic()`, zeroed to first frame |
