import os

def update_labels_in_all_folders(main_folder):
    # 遍歷主資料夾內的所有子資料夾和文件
    for root, dirs, files in os.walk(main_folder):
        for filename in files:
            if filename.endswith(".txt"):  # 只處理 .txt 標籤文件
                file_path = os.path.join(root, filename)
                
                # 讀取標籤文件
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                
                # 修改每一行的類別編號為 0 (代表 boat)
                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        # 保留原始的標註框座標，但將類別改為 0
                        parts[0] = '0'
                        updated_lines.append(" ".join(parts) + "\n")
                
                # 將修改後的標籤內容寫回原始文件
                with open(file_path, 'w') as file:
                    file.writelines(updated_lines)
                
                print(f"已更新: {file_path}")

# 使用範例
main_folder = "C:\\Users\\User\\Pictures\\Pictures\\"  # 請替換成包含所有標籤的主資料夾路徑
update_labels_in_all_folders(main_folder)
