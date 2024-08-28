import cv2
import os


def img2mp4(input_folder, output_video):
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    # 获取文件夹中所有图片文件的路径
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.png')]

    # 对图片文件路径按照数字从小到大排序
    image_paths.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    # 读取第一张图片来获取图片尺寸
    frame = cv2.imread(image_paths[0])
    height, width, _ = frame.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

    # 逐个将图片写入视频
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放资源
    video.release()


if __name__ == '__main__':
    motions = ['swing_dancing']
    names = ['abe', 'brian', 'drake', 'elizabeth', 'james', 'leonard', 'lewis', 'maynard', 'megan', 'remy',
             'sophie', 'shannon']
    # 设置输入文件夹和输出视频文件名
    for motion in motions:
        for name in names:
            input_folder = os.path.join('D:\\pythonProjects\\openpose-skeleton-transformer\\datasets\\img\\seq\\', motion, name)
            output_video = os.path.join('D:\\pythonProjects\\openpose-skeleton-transformer\\datasets\\video\\seq\\', motion, name, 'output.mp4')
            img2mp4(input_folder, output_video)
    # input_folder = os.path.join(
    #     'D:\\pythonProjects\\openpose-skeleton-transformer\\pose3D\\seq')
    # output_video = os.path.join('D:\\pythonProjects\\openpose-skeleton-transformer\\pose3D\\seq', 'seq.mp4')
    # img2mp4(input_folder, output_video)
    # input_folder = os.path.join('D:\\pythonProjects\\openpose-skeleton-transformer\\datasets\\img\\seq\\', 'snatch', 'maynard')
    # output_video = os.path.join('D:\\pythonProjects\\openpose-skeleton-transformer\\datasets\\video\\seq\\', 'snatch', 'maynard', 'output.mp4')
    # img2mp4(input_folder, output_video)
