import pyrealsense2 as rs
import numpy as np
import cv2

def get_depth_frame(pipeline):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None
    return np.asanyarray(depth_frame.get_data(), dtype=np.float32) * 0.001

def compute_tilt_indicator(depth, top_ratio=0.1, bottom_ratio=0.1):
    h, w = depth.shape
    top_rows = depth[:int(h * top_ratio), :]
    bottom_rows = depth[int(h * (1 - bottom_ratio)):, :]
    avg_top = np.nanmean(top_rows)
    avg_bottom = np.nanmean(bottom_rows)
    tilt_diff = avg_top - avg_bottom
    return tilt_diff, avg_top, avg_bottom

def colorize_depth(depth):
    lower, upper = np.percentile(depth, [5, 95])
    depth_clipped = np.clip(depth, lower, upper)
    depth_vis = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = np.uint8(depth_vis)
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    pipeline.start(config)

    try:
        while True:
            depth = get_depth_frame(pipeline)
            if depth is None:
                continue

            h, w = depth.shape
            crop_h_start = int(h * 0.25)
            crop_h_end = int(h * 0.75)
            crop_w_start = int(w * 0.25)
            crop_w_end = int(w * 0.5)
            depth_crop = depth[crop_h_start:crop_h_end, crop_w_start:crop_w_end]

            tilt_diff, avg_top, avg_bottom = compute_tilt_indicator(depth_crop)

            if abs(tilt_diff) < 0.005:
                tilt_text = "Level"
                color = (0, 255, 0)
            elif tilt_diff > 0:
                tilt_text = "Tilt Down"
                color = (0, 255, 255)
            else:
                tilt_text = "Tilt Up"
                color = (0, 0, 255)

            depth_colored = colorize_depth(depth)
            top_line = crop_h_start + int(depth_crop.shape[0]*0.1)
            bottom_line = crop_h_start + int(depth_crop.shape[0]*0.9)
            cv2.line(depth_colored, (crop_w_start, top_line), (crop_w_end, top_line), (0,255,0), 2)
            cv2.line(depth_colored, (crop_w_start, bottom_line), (crop_w_end, bottom_line), (0,255,0), 2)

            cv2.putText(depth_colored, tilt_text, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow('Depth Alignment', depth_colored)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
