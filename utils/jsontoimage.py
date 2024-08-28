import json
import os
import argparse
import cv2
import numpy as np

# 骨骼关键点连接对
pose_pairs = [
    [0, 1], [0, 15], [0, 14],
    [15, 17],
    [14, 16],
    [1, 2], [1, 5], [1, 8], [1, 11],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [8, 9], [9, 10],
    [11, 12], [12, 13],
]

# 绘制用的颜色
pose_colors = [
    (153, 0, 61), (153, 51, 0), (153, 102, 0), (153, 153, 0),
    (102, 153, 0), (51, 153, 0), (0, 153, 0), (0, 153, 51),
    (0, 153, 102), (0, 153, 153), (0, 102, 153), (0, 51, 153),
    (0, 0, 153), (51, 0, 153), (102, 0, 153), (153, 0., 153),
    (153, 0, 102), (153, 0, 51)
]

# 绘制用的颜色
pose_line_colors = {
    '[0, 1]': (0, 0, 153),
    '[0, 15]': (153, 0, 153),
    '[0, 14]': (51, 0, 153),
    '[15, 17]': (153, 0, 102),
    '[14, 16]': (102, 0, 153),
    '[1, 2]': (153, 0, 0),
    '[1, 5]': (153, 51, 0),
    '[1, 8]': (0, 153, 0),
    '[1, 11]': (0, 153, 153),
    '[2, 3]': (153, 102, 0),
    '[3, 4]': (153, 153, 0),
    '[5, 6]': (102, 153, 0),
    '[6, 7]': (51, 153, 0),
    '[8, 9]': (0, 153, 51),
    '[9, 10]': (0, 153, 102),
    '[11, 12]': (0, 102, 153),
    '[12, 13]': (0, 51, 153),
}


def draw_json_absolute(json_file, result_folder):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(json_file, 'r') as f:
        data = json.load(f)
    script_path = os.path.abspath(__file__)

    # 获取脚本所在目录的路径
    script_dir = os.path.dirname(script_path)
    img_path = os.path.join(script_dir, '../datasets/black_256.png')
    img = cv2.imread(img_path)
    # img = cv2.imread('../datasets/black_256.png')

    for d in data['people']:
        '''pose身体部分'''
        kpt = np.array(d['pose_keypoints_2d'])[:28].reshape((14, 2))
        # kpt = kpt[:-4]
        for key, color in pose_line_colors.items():
            # 解析关键点序号
            indices = [int(idx) for idx in key.strip('[]').split(',')]
            idx1, idx2 = indices
            if idx1 >= 14 or idx2 >= 14:
                continue
            # 获取颜色
            color_opencv = (color[2], color[1], color[0])  # OpenCV中颜色格式为BGR

            # 获取关键点坐标
            x1, y1 = kpt[idx1]
            x2, y2 = kpt[idx2]

            # 将相对坐标转换为绝对像素坐标

            if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0:
                continue
            # 将坐标转换为整数
            x1_int, y1_int = int(x1), int(y1)
            x2_int, y2_int = int(x2), int(y2)

            # 绘制线段
            cv2.line(img, (x1_int, y1_int), (x2_int, y2_int), color_opencv, 4)

        for i, (x, y) in enumerate(kpt):
            if i in (14, 15, 16, 17):
                continue
            if x == 0 or y == 0:
                continue

            # 获取颜色
            color = pose_colors[i]

            # 将颜色转换为OpenCV格式
            color_opencv = (color[2], color[1], color[0])  # OpenCV中颜色格式为BGR

            # 将坐标转换为整数
            x_int, y_int = int(x), int(y)

            # 绘制点
            cv2.circle(img, (x_int, y_int), 4, color_opencv, -1, )

    # 保存图片
    basename = os.path.basename(json_file)
    img_name = basename.replace('.json', '.png')
    print('save image {}'.format(os.path.join(result_folder, img_name)))
    # img_name = basename.replace('keypoints.json', 'rendered.png')
    cv2.imwrite(os.path.join(result_folder, img_name), img)



def draw_json_relative(img_file, json_file):
    print('handle json {}'.format(json_file))

    with open(json_file, 'r') as f:
        data = json.load(f)

    # 纯黑色背景
    img = cv2.imread('black_new.jpg')

    for d in data['people']:
        '''pose身体部分'''
        kpt = np.array(d['pose_keypoints_2d']).reshape((18, 2))
        kpt = kpt[:-4]
        for key, color in pose_line_colors.items():
            # 解析关键点序号
            indices = [int(idx) for idx in key.strip('[]').split(',')]
            idx1, idx2 = indices
            if idx1 >= 14 or idx2 >= 14:
                continue
            # 获取颜色
            color_opencv = (color[2], color[1], color[0])  # OpenCV中颜色格式为BGR

            # 获取关键点坐标
            x1, y1 = kpt[idx1]
            x2, y2 = kpt[idx2]

            # 将相对坐标转换为绝对像素坐标
            x1 = x1 * img.shape[1]
            y1 = y1 * img.shape[0]
            x2 = x2 * img.shape[1]
            y2 = y2 * img.shape[0]
            if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0:
                continue
            # 将坐标转换为整数
            x1_int, y1_int = int(x1), int(y1)
            x2_int, y2_int = int(x2), int(y2)

            # 绘制线段
            cv2.line(img, (x1_int, y1_int), (x2_int, y2_int), color_opencv, 4)

        for i, (x, y) in enumerate(kpt):
            if i in (14, 15, 16, 17):
                continue
            if x == 0 or y == 0:
                continue

            # 获取颜色
            color = pose_colors[i]

            # 将颜色转换为OpenCV格式
            color_opencv = (color[2], color[1], color[0])  # OpenCV中颜色格式为BGR
            x = x * img.shape[1]
            y = y * img.shape[0]
            # 将坐标转换为整数
            x_int, y_int = int(x), int(y)

            # 绘制点
            cv2.circle(img, (x_int, y_int), 4, color_opencv, -1, )

        # 保存图片
        img_name = os.path.basename(img_file)
        cv2.imwrite(os.path.join('results/img', img_name), img)


if __name__ == '__main__':
    # # 解析命令行参数
    # parser = argparse.ArgumentParser(description="Process image JSON pairs")
    # parser.add_argument("--json", default='D:\\pythonProjects\\openpose-skeleton-transformer\\datasets\\train\\json\\seq\\standards\\joyful_jump', help="JSON folder path")
    # parser.add_argument("--result", default='D:\\pythonProjects\\openpose-skeleton-transformer\\datasets\\train\\json\\seq\\standards\\test', help="Result folder path")
    # args = parser.parse_args()
    #
    #
    # if not os.path.exists(args.result):
    #     os.makedirs(args.result)
    #
    # # 遍历文件夹
    # for json_file in os.listdir(args.json):
    #     if json_file.endswith('keypoints.json'):
    #         # 构建的完整路径
    #         json_path = os.path.join(args.json, json_file)
    #         # 调用 draw_json_absolute 函数
    #         draw_json_absolute(json_path, args.result)
    draw_json_absolute('D:\\pythonProjects\\openpose-skeleton-transformer\\datasets\\train\\json\\seq\\goalkeeper_catch\\shannon\\022_keypoints.json', 'D:\\pythonProjects\\openpose-skeleton-transformer\\datasets\\train\\img\\seq\\goalkeeper_catch\\shannon')