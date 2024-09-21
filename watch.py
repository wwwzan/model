import cv2

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频时长（秒）
    duration = total_frames / fps

    print(f"视频总帧数: {total_frames}")
    print(f"视频帧率: {fps}")
    print(f"视频时长（秒）: {duration}")

    cap.release()


video_path= r'D:\video\32.31.250.108\20240501_20240501135236_20240501160912_135235.mp4'
get_video_info(video_path)

