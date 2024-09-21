from ultralytics import YOLOv10
from ultralytics.solutions import speed_estimation
import cv2

# 加载YOLOv10模型
model = YOLOv10("yolov10n.pt")
# 获取模型中的对象名称
names = model.model.names

# 打开视频文件
cap = cv2.VideoCapture(r'D:\video\32.31.250.108\20240501_20240501135236_20240501160912_135235.mp4')

# 获取视频的宽度、高度和帧率
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 创建视频写入器，用于输出处理后的视频
video_writer = cv2.VideoWriter("out.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

# 设置测速线段的两个端点
line_pts = [(0, 380), (1640, 380)]

# 初始化速度估计器
speed_obj = speed_estimation.SpeedEstimator()
# 设置速度估计器的参数，包括测速线段、对象名称和是否显示图像
speed_obj.set_args(reg_pts=line_pts,
                   names=names,
                   view_img=True)

# 循环读取视频帧
while cap.isOpened():
    # 读取一帧
    success, im0 = cap.read()
    # 如果读取失败，则退出循环
    if not success:
        break
    tracks = model.track(im0, persist=True, show=False)
    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)

# 释放视频读取器和写入器
cap.release()
video_writer.release()
# 销毁所有OpenCV窗口
cv2.destroyAllWindows()
