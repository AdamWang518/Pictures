import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

# 載入 .mat 檔案
def load_mat_file(file_path):
    return loadmat(file_path)

# 提取座標數據
def extract_locations(data):
    return data['image_info'][0,0][0,0][0]

# 生成密度圖
def generate_density_map(locations, image_shape=(768, 1024), sigma=10):
    density_map = np.zeros(image_shape)
    for location in locations:
        x, y = int(location[0]), int(location[1])
        if x < image_shape[1] and y < image_shape[0]:
            density_map[y, x] += 1
    return gaussian_filter(density_map, sigma=sigma)

# 繪製密度圖
def plot_density_map(density_map):
    plt.figure(figsize=(10, 8))
    plt.imshow(density_map, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.title('Density Map')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(False)
    plt.show()

# 主程式
if __name__ == '__main__':
    file_path = 'D:\\MEGA\\GT_IMG_1.mat'  # 修改此處為您的檔案路徑
    mat_data = load_mat_file(file_path)
    locations = extract_locations(mat_data)
    density_map = generate_density_map(locations)
    plot_density_map(density_map)
