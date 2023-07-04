import cv2
import numpy as np
import torch
from PIL import Image
from model_2 import Net
from torchvision import transforms

img_rows, img_cols = 48, 48
num_classes = 2
batch_size = 512

model = torch.load('D:\\Backlog\pythonProject4\\face_and_emotion_detection-master\\modal\\model_final.pt')
model.eval()

# 定义预处理操作
transform = transforms.Compose([
    transforms.Resize((img_rows, img_cols)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 打开摄像头
cap = cv2.VideoCapture(0)

# 定义计数器，每10帧进行一次图像识别
count = 0

while True:
    ret, frame = cap.read()

    if ret:
        # 对每一帧进行预处理和人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 遍历每个检测到的人脸
        for (x, y, w, h) in faces:
            # 提取人脸图像
            face_img = frame[y:y+h, x:x+w]

            # 对人脸图像进行预处理
            img = Image.fromarray(face_img)
            img = transform(img)
            img = img.unsqueeze(0)

            # 进行情感分析
            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs.data, 1)

            if predicted.item() == 0:
                label = "Eaten"
            else:
                label = "NOT Eaten"

            # 在图像上绘制情感分析结果
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print(label)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示每一帧
        cv2.imshow("frame", frame)

        # 按下 q 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()