from PIL import Image
import numpy as np
from scipy.io import savemat
import os

# 函数：将YOLO格式转换为MAT文件并保存到指定的输出目录
def yolo_to_mat(yolo_file_path, image_path, output_mat_path):
    with Image.open(image_path) as img:
        image_size = img.size  # (width, height)
    with open(yolo_file_path, 'r') as file:
        yolo_data = file.readlines()
    locations = []
    for line in yolo_data:
        _, x_center, y_center, width, height = map(float, line.split())
        abs_x_center = x_center * image_size[0]
        abs_y_center = y_center * image_size[1]
        abs_width = width * image_size[0]
        abs_height = height * image_size[1]
        locations.append([abs_x_center - abs_width / 2, abs_y_center - abs_height / 2,
                          abs_x_center + abs_width / 2, abs_y_center + abs_height / 2])
    mat_data = {
        'image_info': np.array([[(np.array(locations), np.array([[len(locations)]]))]], 
                               dtype=[('location', 'O'), ('number', 'O')])
    }
    savemat(output_mat_path, mat_data)

# 函数：处理特定的图像和标注文件夹，将.mat文件输出到指定路径
def process_yolo_folder(images_folder, labels_folder, output_folder):
    for image_file in os.listdir(images_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(images_folder, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            yolo_file_path = os.path.join(labels_folder, label_file)

            if os.path.exists(yolo_file_path):
                # 构建输出的.mat文件路径，将.mat文件保存到输出目录中
                output_mat_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.mat')

                # 创建输出文件夹（如果不存在）
                os.makedirs(output_folder, exist_ok=True)
                
                # 调用yolo_to_mat进行转换
                yolo_to_mat(yolo_file_path, image_path, output_mat_path)
                print(f"Converted {image_file} and saved .mat file in {output_mat_path}")

# 函数：处理train、valid和test文件夹，将.mat文件输出到指定目录
def process_dataset_folder(root_folder, output_root_folder):
    subsets = ['train', 'valid', 'test']
    for subset in subsets:
        images_folder = os.path.join(root_folder, subset, 'images')
        labels_folder = os.path.join(root_folder, subset, 'labels')

        # 确保images和labels文件夹都存在
        if os.path.exists(images_folder) and os.path.exists(labels_folder):
            print(f"Processing {subset} set...")
            output_folder = os.path.join(output_root_folder, subset)  # 在输出根目录中创建子目录
            process_yolo_folder(images_folder, labels_folder, output_folder)

# 指定根目录和输出目录
dataset_root_folder = 'D:\\MEGA\\DataSet\\Pictures\\'  # 输入文件夹路径
output_root_folder = 'D:\\MEGA\\DataSet\\YoloToMatShip'  # 输出文件夹路径
process_dataset_folder(dataset_root_folder, output_root_folder)
