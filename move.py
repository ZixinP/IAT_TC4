import os
import random
import shutil

# 定义路径
val_dir = './val'
test_dir = './test'

# 确保 test 文件夹存在
os.makedirs(test_dir, exist_ok=True)

# 遍历 val 文件夹中的每个类别子文件夹
for class_name in os.listdir(val_dir):
    class_val_dir = os.path.join(val_dir, class_name)
    class_test_dir = os.path.join(test_dir, class_name)

    # 如果是文件夹，处理该类别
    if os.path.isdir(class_val_dir):
        os.makedirs(class_test_dir, exist_ok=True)  # 确保 test 中的类别文件夹存在

        # 获取该类别文件夹中的所有文件
        files = os.listdir(class_val_dir)
        files_to_move = random.sample(files, len(files) // 3)  # 随机选择 1/3 的文件

        # 移动文件到 test 文件夹
        for file_name in files_to_move:
            src_path = os.path.join(class_val_dir, file_name)
            dest_path = os.path.join(class_test_dir, file_name)
            shutil.move(src_path, dest_path)

print("图片已成功从 val 移动到 test 文件夹！")