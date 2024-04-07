import time
import myUtils
import cv2
from ultralytics import YOLO

# 初始化 YOLO 模型
model = YOLO(r'D:\PythonProject\ultralytics-main\mycode\pokeTrains\train4\weights\best.pt')

# 实时捕获窗口截图并显示识别结果
while True:
    # 获取窗口截图
    screenshot = myUtils.main('欢乐斗地主')

    # 使用 YOLO 模型进行窗口识别处理
    img_results = model(screenshot, conf=0.5)

    # 获取处理结果图像
    img_result = img_results[0].plot(
        conf=True,
        line_width=1
    )

    # 在窗口中显示识别结果
    cv2.imshow('Window Detection', img_result)

    # 检测按键 'q' 是否被按下，如果按下则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(1)
# 关闭窗口并释放资源
cv2.destroyAllWindows()
