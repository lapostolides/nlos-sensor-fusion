# DW1000 Driver SPI Changes

Documents modifications to the Decawave DW1000 driver relative to the upstream
[dwm1001-examples](https://github.com/Decawave/dwm1001-examples) repository.
Two files were changed; the changes are tightly coupled.

---

## Problem: VLA Stack Overflow on CIR Reads

The upstream `readfromspi` and `writetospi` in `port_platform.c` allocate their
SPI transfer buffers on the stack as C99 variable-length arrays (VLAs):

```c
// upstream original
uint8 idatabuf[idatalength];   // stack — size unknown at compile time
uint8 itempbuf[idatalength];
```

For normal register reads (a few bytes) this is fine. But reading the DW1000
CIR accumulator (`dwt_readaccdata`) requires a single logical transfer of
**4065 bytes** (1016 samples × 4 bytes + 1 dummy byte). Two VLAs of that size
would put **~8 KB** on the FreeRTOS task stack, reliably causing a stack
overflow on the `"SSTWR_RESP"` task (which only allocates 10 000 bytes total
and already holds other locals and frame overhead).

Additionally, the nRF52832 SPIM peripheral (EasyDMA mode) has a **255-byte
hardware limit** per DMA transfer. A single 4065-byte `nrf_drv_spi_transfer`
call cannot be fulfilled by the hardware regardless of stack space.

---

## Fix: Two Coupled Changes

### 1. `port_platform.c` — Static SPI Buffers

`readfromspi` and `writetospi` were changed to use static fixed-size buffers
instead of VLAs, with an explicit bounds guard:

```c
#define SPI_BUF_SIZE 260

static uint8 spi_tx[SPI_BUF_SIZE];
static uint8 spi_rx[SPI_BUF_SIZE];

int readfromspi(uint16 headerLength, const uint8 *headerBuffer,
                uint32 readlength,   uint8 *readBuffer)
{
    uint32 idatalength = headerLength + readlength;

    if (idatalength > SPI_BUF_SIZE)   // hard guard — never overrun
        return -1;

    memset(spi_tx, 0, idatalength);
    memset(spi_rx, 0, idatalength);
    memcpy(spi_tx, headerBuffer, headerLength);

    spi_xfer_done = false;
    nrf_drv_spi_transfer(&spi, spi_tx, idatalength, spi_rx, idatalength);
    while (!spi_xfer_done) ;

    memcpy(readBuffer, spi_rx + headerLength, readlength);
    return 0;
}
```

`writetospi` follows the same pattern (static buffers, same `SPI_BUF_SIZE`
guard).

**Why 260 bytes?**
The largest single SPI call after chunking is `ACC_SPI_CHUNK + 1 + header`:

| Component | Bytes |
|---|---|
| SPI sub-register header (ACC_MEM_ID) | 3 |
| CIR dummy byte (always prepended by DW1000) | 1 |
| CIR data per chunk (`ACC_SPI_CHUNK`) | 240 |
| **Total** | **244** |

260 bytes provides ~16 bytes of headroom over the worst-case transfer and stays
comfortably under the 255-byte SPIM EasyDMA hardware limit.

---

### 2. `deca_device.c` — Chunked Accumulator Read

Because `readfromspi` now rejects any transfer larger than 260 bytes, the
upstream `dwt_readaccdata` (a single call to `dwt_readfromdevice` for the full
4065 bytes) would return -1 and produce an empty buffer. The function was
rewritten to loop in 240-byte chunks:

```c
#define ACC_SPI_CHUNK 240

void dwt_readaccdata(uint8 *buffer, uint16 len, uint16 accOffset)
{
    _dwt_enableclocks(READ_ACC_ON);

    while (len > 0) {
        uint16 n = (len > ACC_SPI_CHUNK) ? (ACC_SPI_CHUNK + 1) : (len + 1);
        dwt_readfromdevice(ACC_MEM_ID, accOffset, n, buffer);
        if (accOffset == 0) {
            buffer    += n;
            accOffset += n - 1;
            len       -= n - 1;
        } else {
            memmove(buffer, buffer + 1, (size_t)(n - 1));
            buffer    += n - 1;
            accOffset += n - 1;
            len       -= n - 1;
        }
    }

    _dwt_enableclocks(READ_ACC_OFF);
}
```

**Why `n = chunk + 1`?**
The DW1000 always prepends one dummy byte to accumulator data regardless of
the start offset (documented in the DW1000 User Manual). The extra `+1` in
every read request absorbs this dummy byte into the transfer so the loop's
bookkeeping stays aligned: after each chunk, `buffer` and `accOffset` both
advance by `n - 1` (the real data bytes, not the dummy).

**Special case for `accOffset == 0` (first chunk):**
On the very first read the caller's buffer pointer is used directly without
`memmove`, because `ss_resp_main.c` already expects the dummy byte at index 0
(`cir_buf[0]`) and skips it when transmitting:

```c
// ss_resp_main.c — cir_buf[0] is the dummy; real IQ starts at cir_buf[1]
for (i = 0; i < CIR_BUF_BYTES; i++)
    SEND_XOR(cir_buf[i + 1]);
```

Subsequent chunks (where `accOffset > 0`) use `memmove` to shift the dummy
byte out of the buffer in-place before advancing the pointer.

---

## Transfer Budget for a Full CIR Read

`dwt_readaccdata(cir_buf, 4065, 0)` makes 18 SPI transactions:

| Chunk | `n` requested | Header bytes | Total SPI bytes | Data retained |
|---|---|---|---|---|
| 1–17 | 241 | 3 | 244 | 240 each |
| 18 (remainder: 4065 − 17×240 = 225) | 226 | 3 | 229 | 225 |

All transactions ≤ 244 bytes — within both the 260-byte static buffer and the
255-byte SPIM EasyDMA hardware limit.

---

## RAM Impact

| Symbol | Location | Size |
|---|---|---|
| `spi_tx[]` | BSS (static, `port_platform.c`) | 260 bytes |
| `spi_rx[]` | BSS (static, `port_platform.c`) | 260 bytes |
| **Total** | | **520 bytes** |

The upstream approach used zero static RAM for SPI buffers but placed up to
~8 KB on the task stack for CIR reads. The static allocation trades a fixed
520 bytes of BSS for elimination of the stack overflow risk.
