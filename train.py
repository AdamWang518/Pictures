import torch
from ultralytics import YOLO

def main():
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 YOLOv10n 模型（使用预训练权重）
    model = YOLO('yolov10n.pt').to(device)

    # 配置训练参数
    data_path = 'data.yaml'  # 请替换成你的数据集配置文件路径
    epochs = 100  # 训练的轮数，可以根据需要调整
    batch = 32  # 批次大小
    img_size = 640  # 图片大小
    learning_rate = 0.01  # 学习率

    # 开始训练
    model.train(data=data_path, 
                epochs=epochs, 
                batch=batch,  # 将 batch_size 修改为 batch
                imgsz=img_size,
                lr0=learning_rate)

    # 保存训练后的模型
    model.save('best_yolov10n_model.pt')

    # 输出训练完成的提示
    print("训练完成，模型已保存为 'best_yolov10n_model.pt'")

# 确保在 Windows 下使用多进程时正常工作
if __name__ == '__main__':
    main()
