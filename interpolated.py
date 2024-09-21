import cv2

# 定义补帧函数
def interpolate_frames(video_file, target_fps):
    video = cv2.VideoCapture(video_file)

    original_fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = video_file.replace('.mp4', '_interpolated.mp4')
    out = cv2.VideoWriter(output_file, fourcc, target_fps, (width, height))

    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    # 插值处理
    for i in range(len(frames) - 1):
        out.write(frames[i])  # 写入当前帧
        num_interpolated_frames = int(target_fps / original_fps) - 1
        for j in range(1, num_interpolated_frames + 1):
            alpha = j / (num_interpolated_frames + 1)
            interpolated_frame = cv2.addWeighted(frames[i], 1 - alpha, frames[i + 1], alpha, 0)
            out.write(interpolated_frame)

    out.write(frames[-1])  # 写入最后一帧

    video.release()
    out.release()

    return output_file

# 文件路径
video_file = 'fourth_combined.mp4'

# 设定原始帧率和目标帧率
original_fps = 25  # 原始帧率
target_fps = original_fps * 10  # 目标帧率为250

# 进行十倍插帧处理
interpolated_video_file = interpolate_frames(video_file, target_fps)

print(f"插帧后的视频已保存为: {interpolated_video_file}")
