import torch
from ultralytics import YOLO
import cv2
import os
import numpy as np

def slice_image(image, slice_size=640):
    """
    将图片切割成指定大小的块
    """
    height, width, _ = image.shape
    slices = []
    coords = []

    # 计算在图片宽度和高度方向上的切片数量
    x_slices = range(0, width, slice_size)
    y_slices = range(0, height, slice_size)

    for y in y_slices:
        for x in x_slices:
            x_max = min(x + slice_size, width)
            y_max = min(y + slice_size, height)
            slice_img = image[y:y_max, x:x_max]
            slices.append(slice_img)
            coords.append((x, y))
    return slices, coords

def merge_results(image, results, coords, slice_size=640):
    """
    将检测结果合并回原始图片坐标系
    """
    combined_boxes = []
    combined_scores = []
    combined_classes = []

    for i, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu().numpy()  # 获取检测框坐标
        scores = result.boxes.conf.cpu().numpy()  # 获取置信度
        classes = result.boxes.cls.cpu().numpy()  # 获取类别

        x_offset, y_offset = coords[i]
        for box in boxes:
            box[0] += x_offset  # xmin
            box[1] += y_offset  # ymin
            box[2] += x_offset  # xmax
            box[3] += y_offset  # ymax
        combined_boxes.append(boxes)
        combined_scores.append(scores)
        combined_classes.append(classes)

    # 将所有检测结果拼接在一起
    if combined_boxes:
        combined_boxes = np.vstack(combined_boxes)
        combined_scores = np.hstack(combined_scores)
        combined_classes = np.hstack(combined_classes)
    else:
        combined_boxes = np.array([])
        combined_scores = np.array([])
        combined_classes = np.array([])

    return combined_boxes, combined_scores, combined_classes

def plot_boxes_on_image(image, boxes, scores, classes, class_names):
    """
    在原始图片上绘制检测框
    """
    for box, score, cls in zip(boxes, scores, classes):
        xmin, ymin, xmax, ymax = map(int, box)
        label = f"{class_names[int(cls)]} {score:.2f}"
        # 绘制矩形框
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
        # 绘制标签
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, color=(0, 255, 0), thickness=1)
    return image

def main():
    # 检查 GPU 可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model = YOLO('best_10n_final.pt')  # 替换为你的模型路径
    model.to(device)

    # 类别名称（根据你的数据集进行修改）
    class_names = ['boat']  # 替换为你的类别名称列表

    # 输入和输出文件夹路径
    input_folder = 'D:\\MEGA\\DataSet\\randomPick\\0811'  # 替换为你的输入图片文件夹路径
    output_folder = 'D:\\MEGA\\DataSet\\ShipsSliceOut\\0811newTest'  # 替换为你的输出文件夹路径

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):  # 只处理图片文件
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取图片
            img = cv2.imread(input_path)
            if img is None:
                print(f"无法读取图片 {input_path}")
                continue

            # 将图片切割成小块
            slices, coords = slice_image(img, slice_size=640)

            # 对每个切片进行目标检测
            results = []
            for slice_img in slices:
                # 进行目标检测
                result = model(slice_img, device=device)[0]
                results.append(result)

            # 合并检测结果
            boxes, scores, classes = merge_results(img, results, coords, slice_size=640)

            # 如果有检测结果，绘制检测框
            if boxes.size > 0:
                annotated_img = plot_boxes_on_image(img, boxes, scores, classes, class_names)
            else:
                annotated_img = img  # 如果没有检测结果，原图保存

            # 保存带有检测框的图片到输出文件夹
            cv2.imwrite(output_path, annotated_img)
            print(f"已保存检测结果到 {output_path}")

if __name__ == '__main__':
    main()
