import numpy as np, os
dm_dir = 'C:/Users/lapos/nlos-sensor-fusion/data/testlaserroom/colmap_output/dense/stereo/depth_maps'
files = sorted(f for f in os.listdir(dm_dir) if f.endswith('.geometric.bin'))
print(f"Found {len(files)} geometric depth maps")
for fn in files[:5]:
    data = open(os.path.join(dm_dir, fn), 'rb').read()
    amp = [i for i,b in enumerate(data[:50]) if b == 38]
    end = amp[2]+1
    d = np.frombuffer(data[end:], dtype=np.float32)
    valid = d[d > 0]
    if len(valid) > 0:
        print(f'{fn[:35]}: valid={100*len(valid)/len(d):.1f}%, median={np.median(valid):.2f}m')
    else:
        print(f'{fn[:35]}: ALL ZEROS')
