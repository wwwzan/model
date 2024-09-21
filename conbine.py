import cv2

# 定义视频拼接函数
def stitch_videos(video_files, output_file):
    video_objects = [cv2.VideoCapture(file) for file in video_files]

    # 获取第一个视频的参数
    fps = int(video_objects[0].get(cv2.CAP_PROP_FPS))
    frame_width = int(video_objects[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_objects[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    for video in video_objects:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            out.write(frame)  # 写入当前帧
        video.release()  # 释放当前视频对象

    out.release()  # 释放视频写入对象

# 视频文件路径
fourth_video_files = [
    r'D:\video\32.31.250.108\20240501_20240501135236_20240501160912_135235.mp4',
    r'D:\video\32.31.250.108\20240501_20240501135236_20240501160912_135235.mp4'
]

# 合并视频
stitch_videos(fourth_video_files, 'fourth_combined.mp4')
