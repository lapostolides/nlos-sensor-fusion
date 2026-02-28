import ctypes, os
env_bin = 'C:/Users/lapos/miniconda3/envs/sensor-fusion/Library/bin'
exe = os.path.join(env_bin, 'colmap.exe')
try:
    h = ctypes.WinDLL(exe)
    print('loaded OK')
except OSError as e:
    print('error:', e)
