import os
from PIL import Image
import numpy as np
from scipy.io import savemat
import shutil

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

# 函数：处理图像和标注文件夹，将.mat文件和图片按序号重命名并移动到指定的目标文件夹
def process_yolo_folder(images_folder, labels_folder, output_images_folder, output_groundtruth_folder, start_index=1):
    count = start_index
    for image_file in os.listdir(images_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(images_folder, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            yolo_file_path = os.path.join(labels_folder, label_file)

            if os.path.exists(yolo_file_path):
                # 构建输出的.mat文件路径，将.mat文件保存到output_groundtruth_folder中
                output_mat_path = os.path.join(output_groundtruth_folder, f"GT_{count}.mat")
                output_image_path = os.path.join(output_images_folder, f"{count}.jpg")
                output_label_path = os.path.join(output_groundtruth_folder, f"{count}.txt")  # 将.txt文件放入ground_truth文件夹

                # 创建输出文件夹（如果不存在）
                os.makedirs(output_images_folder, exist_ok=True)
                os.makedirs(output_groundtruth_folder, exist_ok=True)

                # 将图片转换为 .jpg 格式并保存
                img = Image.open(image_path).convert('RGB')  # 确保统一为 RGB
                img.save(output_image_path, "JPEG")
                
                # 将对应的txt文件复制到ground_truth文件夹，并重命名为相同序号
                shutil.copy(yolo_file_path, output_label_path)

                # 调用yolo_to_mat进行转换并保存.mat文件
                yolo_to_mat(yolo_file_path, image_path, output_mat_path)

                print(f"Moved and renamed {image_file} to {output_image_path}")
                print(f"Generated {output_mat_path}")
                print(f"Copied and renamed {label_file} to {output_label_path}")

                count += 1
    return count

# 函数：处理train、val和test文件夹，按要求将文件输出到指定目录
def process_dataset_folder(root_folder, train_output_folder, test_output_folder):
    subsets = ['train', 'val', 'test']
    current_index = 1

    for subset in subsets:
        images_folder = os.path.join(root_folder, subset, 'images')
        labels_folder = os.path.join(root_folder, subset, 'labels')

        if os.path.exists(images_folder) and os.path.exists(labels_folder):
            if subset == 'train':
                output_images_folder = os.path.join(train_output_folder, 'images')
                output_groundtruth_folder = os.path.join(train_output_folder, 'ground_truth')
            else:
                output_images_folder = os.path.join(test_output_folder, 'images')
                output_groundtruth_folder = os.path.join(test_output_folder, 'ground_truth')

            print(f"Processing {subset} set...")
            current_index = process_yolo_folder(images_folder, labels_folder, output_images_folder, output_groundtruth_folder, current_index)

# 指定根目录和输出目录
dataset_root_folder = 'D:\\Github\\Pictures\\'  # 输入文件夹路径
train_output_folder = 'D:\\MEGA\\DataSet\\YoloToMatShipRevise\\train_data'  # Train数据输出路径
test_output_folder = 'D:\\MEGA\\DataSet\\YoloToMatShipRevise\\test_data'  # Test数据输出路径

process_dataset_folder(dataset_root_folder, train_output_folder, test_output_folder)
