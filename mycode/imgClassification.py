# 做分类
import cv2
from ultralytics import YOLO

# 初始化YOLOv5模型，加载预训练权重
model = YOLO('../yolov8n-cls.pt')

# 通过模型检测图像中的对象
img = model('../mymedia/img.png')  # 通过模型处理图像

# 绘制检测结果
imgResult = img[0].plot()  # 获取并绘制检测结果

# 使用OpenCV显示结果图像
cv2.imshow('imgResult', imgResult)  # 在窗口中显示结果图像
cv2.waitKey(0)  # 等待按键响应，0 表示一直等待直到有按键被按下
