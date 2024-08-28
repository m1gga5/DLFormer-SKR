import os
import cv2

# 设置图片的根目录
image_root_dir = r"D:\pythonProjects\openpose-skeleton-editing\datasets\img\seq"

# 遍历根目录及其子目录中的所有文件
for subdir, dirs, files in os.walk(image_root_dir):
    for file in files:
        # 构建完整的文件路径
        file_path = os.path.join(subdir, file)

        # 确保文件是图片
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            # 使用cv2读取图片
            img = cv2.imread(file_path)

            if img is not None:
                # Resize图片为256x256
                img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

                # 保存修改后的图片，覆盖原文件
                cv2.imwrite(file_path, img_resized)

print("图片处理完成。")