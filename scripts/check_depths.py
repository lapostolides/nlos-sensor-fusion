import numpy as np
from pathlib import Path

base = Path('C:/Users/lapos/nlos-sensor-fusion/data/testlaserroom/colmap_output/dense/stereo/depth_maps')

for kind in ['photometric', 'geometric']:
    maps = sorted(base.glob(f'*.{kind}.bin'))
    print(f"\n--- {kind} ({len(maps)} maps) ---")
    for p in maps[:5]:
        raw = p.read_bytes()
        header_end = raw.index(b'&', raw.index(b'&', raw.index(b'&') + 1) + 1) + 1
        w, h, _ = (int(x) for x in raw[:header_end].decode().strip('&').split('&'))
        d = np.frombuffer(raw[header_end:], dtype=np.float32).reshape(h, w)
        valid = d[d > 0]
        if len(valid):
            print(f'  {p.name}: valid={len(valid)/d.size*100:.0f}%  '
                  f'min={valid.min():.2f}  mean={valid.mean():.2f}  p90={np.percentile(valid,90):.2f}  max={valid.max():.1f}m')
        else:
            print(f'  {p.name}: ALL ZEROS')
