import numpy as np
from pathlib import Path

base = Path('C:/Users/lapos/nlos-sensor-fusion/data/testlaserroom/colmap_output/dense/stereo/depth_maps')

for kind in ['photometric', 'geometric']:
    p = sorted(base.glob(f'*.{kind}.bin'))[0]
    raw = p.read_bytes()
    header_end = raw.index(b'&', raw.index(b'&', raw.index(b'&') + 1) + 1) + 1
    w, h, _ = (int(x) for x in raw[:header_end].decode().strip('&').split('&'))
    d = np.frombuffer(raw[header_end:], dtype=np.float32).reshape(h, w)
    valid = d[d > 0]
    if len(valid):
        print(f'{kind} ({p.name}): {w}x{h}  valid={len(valid)/d.size*100:.1f}%  '
              f'min={valid.min():.3f}  max={valid.max():.3f}  mean={valid.mean():.3f}  '
              f'p10={np.percentile(valid,10):.2f}  p90={np.percentile(valid,90):.2f}')
    else:
        print(f'{kind} ({p.name}): ALL ZEROS')
