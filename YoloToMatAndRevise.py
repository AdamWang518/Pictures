import os
import random
from PIL import Image
import numpy as np
from scipy.io import savemat
import shutil

def yolo_to_mat(yolo_file_path, image_path, output_mat_path):
    """
    Convert YOLO annotation format to the specified MAT format, ensuring consistency with the provided format.
    
    Parameters:
    - yolo_file_path: Path to the YOLO format .txt file.
    - image_path: Path to the image file (to read its size).
    - output_mat_path: Path to save the output MAT file.
    """
    # Read the image to get its size
    with Image.open(image_path) as img:
        image_size = img.size  # (width, height)

    # Read YOLO annotations
    with open(yolo_file_path, 'r') as file:
        yolo_data = file.readlines()

    # Extract point locations (center points) and convert relative coordinates to absolute
    locations = []
    for line in yolo_data:
        class_id, x_center, y_center, width, height = map(float, line.split())
        abs_x_center = x_center * image_size[0]
        abs_y_center = y_center * image_size[1]
        locations.append([abs_x_center, abs_y_center])

    # Convert the locations to a numpy array with dtype float32
    locations = np.array(locations, dtype='float32')

    # Ensure correct data structuring for MAT file
    number_array = np.array([len(locations)], dtype=np.uint8)

    # Create the nested structure with the three layers and correct dtype
    struct_array = np.array([[(locations, number_array)]],
                            dtype=[('location', 'O'), ('number', 'O')])
    image_info = np.empty((1, 1), dtype=object)
    image_info[0, 0] = struct_array

    # Save to MAT file with proper structure
    mat_data = {'image_info': image_info}
    savemat(output_mat_path, mat_data)



# 函数：处理单个文件，并将图片、txt和.mat文件重命名并移动到目标文件夹
def process_single_file(image_file, images_folder, labels_folder, output_images_folder, output_groundtruth_folder, count):
    image_path = os.path.join(images_folder, image_file)
    label_file = os.path.splitext(image_file)[0] + '.txt'
    yolo_file_path = os.path.join(labels_folder, label_file)

    if os.path.exists(yolo_file_path):
        # 构建输出的.mat文件路径，将.mat文件保存到output_groundtruth_folder中
        output_mat_path = os.path.join(output_groundtruth_folder, f"GT_{count}.mat")
        output_image_path = os.path.join(output_images_folder, f"{count}.jpg")
        output_label_path = os.path.join(output_groundtruth_folder, f"{count}.txt")  # 对应的txt文件的目标路径

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

# 函数：处理图像和标注文件夹，将20%数据分配到test，80%数据分配到train
def process_yolo_folder(images_folder, labels_folder, output_train_images_folder, output_train_groundtruth_folder, 
                        output_test_images_folder, output_test_groundtruth_folder, split_ratio=0.8):
    # 获取所有图片文件
    image_files = [f for f in os.listdir(images_folder) if f.endswith((".jpg", ".png"))]
    
    # 打乱文件顺序并划分80%作为训练，20%作为测试
    random.shuffle(image_files)
    split_point = int(len(image_files) * split_ratio)
    train_files = image_files[:split_point]
    test_files = image_files[split_point:]
    
    count_train, count_test = 1, 1  # 训练集和测试集的计数器
    
    # 处理训练集文件
    for image_file in train_files:
        process_single_file(image_file, images_folder, labels_folder, output_train_images_folder, output_train_groundtruth_folder, count_train)
        count_train += 1

    # 处理测试集文件
    for image_file in test_files:
        process_single_file(image_file, images_folder, labels_folder, output_test_images_folder, output_test_groundtruth_folder, count_test)
        count_test += 1

# 函数：按要求将文件输出到指定目录，自动按8:2划分train和test
def process_dataset_folder(root_folder, train_output_folder, test_output_folder, split_ratio=0.8):
    # 汇总所有图片和标注文件
    images_folders = []
    labels_folders = []
    
    subsets = ['train', 'val', 'test']
    
    # 收集所有图片和标签文件夹路径
    for subset in subsets:
        images_folders.append(os.path.join(root_folder, subset, 'images'))
        labels_folders.append(os.path.join(root_folder, subset, 'labels'))

    for images_folder, labels_folder in zip(images_folders, labels_folders):
        if os.path.exists(images_folder) and os.path.exists(labels_folder):
            output_train_images_folder = os.path.join(train_output_folder, 'images')
            output_train_groundtruth_folder = os.path.join(train_output_folder, 'ground_truth')
            output_test_images_folder = os.path.join(test_output_folder, 'images')
            output_test_groundtruth_folder = os.path.join(test_output_folder, 'ground_truth')

            print(f"Processing dataset with {split_ratio*100}% for train and {100-split_ratio*100}% for test...")
            process_yolo_folder(images_folder, labels_folder, output_train_images_folder, output_train_groundtruth_folder, 
                                output_test_images_folder, output_test_groundtruth_folder, split_ratio)

# 指定根目录和输出目录
dataset_root_folder = 'D:\\Github\\Pictures\\'  # 输入文件夹路径
train_output_folder = 'D:\\MEGA\\DataSet\\YoloToMatShipRevise\\train_data'  # Train数据输出路径
test_output_folder = 'D:\\MEGA\\DataSet\\YoloToMatShipRevise\\test_data'  # Test数据输出路径

# 开始处理，将80%文件分到train，20%文件分到test
process_dataset_folder(dataset_root_folder, train_output_folder, test_output_folder, split_ratio=0.8)
