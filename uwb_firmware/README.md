# uwb_firmware

Custom UWB firmware for the DWM1001-DEV board (DW1000 + nRF52832), targeting CIR (Channel Impulse Response) capture for NLOS sensing. Built on the Decawave DW1000 API v2.04 and Nordic nRF5 SDK 14.2, compiled with Segger Embedded Studio.

## Hardware

- **Module**: Qorvo/Decawave DWM1001C (DW1000 UWB IC + nRF52832 SoC)
- **Dev board**: DWM1001-DEV (integrated J-Link, USB-UART bridge)
- **Interface**: UART via J-Link USB, 115200 baud, no flow control

## Examples

| Directory | Role | Output |
|---|---|---|
| `ss_twr_resp/` | Receiver — listens for any UWB poll, reads full CIR accumulator, streams binary frame over UART | 4078-byte binary CIR frames |
| `ss_twr_init_ts/` | Initiator — sends UWB poll at a fixed interval, streams TX timestamp over UART | 10-byte binary timestamp frames |
| `ss_twr_resp_original/` | Original Decawave SS-TWR responder (sends ranging reply, computes distance) | ASCII ranging output |
| `ss_twr_init/` | Original Decawave SS-TWR initiator | ASCII ranging output |
| `ss_twr_init_int/` | Interrupt-driven variant of ss_twr_init | ASCII ranging output |
| `twi_accel/` | LIS2DH12 accelerometer demo via TWI | ASCII accel output |

The primary pair for CIR capture is **`ss_twr_resp`** + **`ss_twr_init_ts`**. See their protocol docs:

- [`examples/ss_twr_resp/receiver-protocol.md`](examples/ss_twr_resp/receiver-protocol.md)
- `examples/ss_twr_init_ts/initiator-protocol.md` (TBD)

## Repository Layout

```
uwb_firmware/
├── boards/
│   └── dw1001_dev.h          # GPIO pin assignments for DWM1001-DEV
├── deca_driver/              # DW1000 API v2.04 (DecaWave)
│   ├── deca_device_api.h
│   ├── deca_regs.h
│   ├── deca_param_types.h
│   ├── deca_params_init.c
│   ├── deca_range_tables.c
│   └── port/port_platform.h  # Antenna delay constants, SPI/UART pin defs
├── examples/
│   ├── ss_twr_resp/          # CIR receiver (active)
│   ├── ss_twr_init_ts/       # TX-timestamp initiator (active)
│   ├── ss_twr_resp_original/ # Original Decawave responder
│   ├── ss_twr_init/          # Original Decawave initiator
│   ├── ss_twr_init_int/      # Interrupt-driven initiator
│   └── twi_accel/            # Accelerometer demo
└── README.md
```

## Building

Open the `.emProject` file in Segger Embedded Studio. Each example has its project under `SES/`.

Required SES packages (install via Tools > Package Manager):
- CMSIS 5 CMSIS-CORE Support Package (v5.02)
- CMSIS-CORE Support Package (v4.05)
- Nordic Semiconductor nRF CPU Support Package (v1.06)

Flash via the integrated J-Link on the DWM1001-DEV.

## Host-Side Capture

[`capture_uwb.py`](../capture_uwb.py) reads binary frames from both boards simultaneously:

```
python capture_uwb.py                           # auto-detect JLink ports
python capture_uwb.py --resp-port COM5 --init-port COM6
python capture_uwb.py --name my-experiment
```

Output is saved to `data/uwb/logs/<run-name>/resp.npz` and `init.npz`.
