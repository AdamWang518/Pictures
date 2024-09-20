import os
import cv2
import torch
import shutil
from ultralytics import YOLO

def slice_image_and_labels(image_path, label_path, output_img_dir, output_lbl_dir, slice_size=640):
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片 {image_path}")
        return

    height, width, _ = img.shape

    # 读取标注文件（YOLO 格式）
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            labels = f.readlines()

    # 将图片切割为小块
    for i in range(0, height, slice_size):
        for j in range(0, width, slice_size):
            # 定义切割区域
            x_min = j
            y_min = i
            x_max = min(j + slice_size, width)
            y_max = min(i + slice_size, height)
            slice_width = x_max - x_min
            slice_height = y_max - y_min

            # 切割图片
            slice_img = img[y_min:y_max, x_min:x_max]

            # 初始化一个列表来存储当前切片的标注
            slice_labels = []

            # 对每个标注进行检查
            for label in labels:
                cls, x, y, w, h = map(float, label.strip().split())
                # 计算原始图像中的实际坐标
                x_center = x * width
                y_center = y * height
                obj_width = w * width
                obj_height = h * height

                # 判断标注是否在当前切割区域内
                if (x_center >= x_min) and (x_center <= x_max) and (y_center >= y_min) and (y_center <= y_max):
                    # 转换为当前切割区域的相对坐标
                    new_x_center = (x_center - x_min) / slice_width
                    new_y_center = (y_center - y_min) / slice_height
                    new_width = obj_width / slice_width
                    new_height = obj_height / slice_height

                    # 确保坐标在 [0,1] 范围内
                    if 0 <= new_x_center <= 1 and 0 <= new_y_center <= 1 and new_width > 0 and new_height > 0:
                        slice_labels.append(f"{int(cls)} {new_x_center} {new_y_center} {new_width} {new_height}\n")

            # 如果当前切片有标注，保存图片和标签文件
            if slice_labels:
                # 保存切割后的图片
                img_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}_{j}.jpg"
                output_img_path = os.path.join(output_img_dir, img_name)
                cv2.imwrite(output_img_path, slice_img)

                # 保存对应的标签文件
                output_label_path = os.path.join(output_lbl_dir, f"{os.path.splitext(img_name)[0]}.txt")
                with open(output_label_path, 'w') as out_lbl:
                    out_lbl.writelines(slice_labels)

            else:
                # 如果没有标注且你不想保留无标注的背景图片，可以选择不保存该图片
                continue  # 不执行任何操作，相当于丢弃该切片

def prepare_dataset(input_img_dir, input_lbl_dir, output_dir, slice_size=640):
    # 定义输出的 images 和 labels 目录
    output_img_dir = os.path.join(output_dir, 'images')
    output_lbl_dir = os.path.join(output_dir, 'labels')

    # 清空输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_img_dir)
    os.makedirs(output_lbl_dir)

    # 遍历所有图片和标签
    for img_file in os.listdir(input_img_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(input_img_dir, img_file)
            label_path = os.path.join(input_lbl_dir, f"{os.path.splitext(img_file)[0]}.txt")
            slice_image_and_labels(image_path, label_path, output_img_dir, output_lbl_dir, slice_size)

def main():
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 配置路径
    input_train_img_dir = 'D:/Github/Pictures/train/images'  # 原始训练图片路径
    input_train_lbl_dir = 'D:/Github/Pictures/train/labels'  # 原始训练标签路径
    input_val_img_dir = 'D:/Github/Pictures/valid/images'      # 原始验证图片路径
    input_val_lbl_dir = 'D:/Github/Pictures/valid/labels'      # 原始验证标签路径

    # 输出目录
    output_train_dir = 'D:/Github/Pictures/sliced/train'  # 切割后的训练数据存放目录
    output_val_dir = 'D:/Github/Pictures/sliced/valid'      # 切割后的验证数据存放目录
    
    # 准备训练集
    print("正在准备训练集...")
    prepare_dataset(input_train_img_dir, input_train_lbl_dir, output_train_dir, slice_size=640)
    # 准备验证集
    print("正在准备验证集...")
    prepare_dataset(input_val_img_dir, input_val_lbl_dir, output_val_dir, slice_size=640)
# 确保在 Windows 下使用多进程时正常工作
if __name__ == '__main__':
    main()