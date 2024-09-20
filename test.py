import torch
from ultralytics import YOLO
import cv2
import os

# 检查 GPU 可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = YOLO('best.pt')  

# 将模型移动到 GPU
model.to(device)

# 输入和输出文件夹路径
input_folder = 'D:\\MEGA\\DataSet\\Ships\\renamed_ships'  # 替换为你的输入图片文件夹路径
output_folder = 'D:\\MEGA\\DataSet\\ShipsOut'  # 替换为你的输出文件夹路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):  # 只处理图片文件
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 读取图片
        img = cv2.imread(input_path)
        if img is None:
            print(f"无法读取图片 {input_path}")
            continue

        # 进行目标检测
        results = model(img)
        # 使用图片原始尺寸推理
        # results = model(img)  # 传入图片的宽和高

        # 绘制检测结果到图片上
        annotated_img = results[0].plot()

        # 保存带有检测框的图片到输出文件夹
        cv2.imwrite(output_path, annotated_img)
        print(f"已保存检测结果到 {output_path}")
