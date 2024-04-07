# 姿态检测
import cv2
from ultralytics import YOLO
import scipy

# 初始化YOLOv5模型，加载预训练权重
model = YOLO('../yolov8n-pose.pt')

# 打开视频文件
capFrames = cv2.VideoCapture('../mymedia/2.mp4')

# 循环读取视频帧
while capFrames.isOpened():
    # 读取视频帧
    successTags, frame = capFrames.read()

    if successTags:
        # 使用模型检测视频帧中的对象
        results = model(frame)

        # 绘制检测结果并显示
        results_proc_show = results[0].plot()
        cv2.imshow('results_proc_show', results_proc_show)  # 在窗口中显示结果图像

        # 检测按键是否为 "q"，如果是则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# 释放视频对象
capFrames.release()

# 关闭所有的OpenCV窗口
cv2.destroyAllWindows()
