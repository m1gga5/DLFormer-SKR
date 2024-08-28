import os

import cv2
import numpy as np
from matplotlib import pyplot as plt, gridspec

from utils.camera import camera_to_world


def img2mp4(path, name):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]

    # 对图片文件路径按照数字从小到大排序
    image_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
    # 读取第一张图片来获取图片尺寸
    frame = cv2.imread(image_paths[0])
    height, width, _ = frame.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(name, fourcc, 30, (width, height))

    # 逐个将图片写入视频
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        video.write(frame)
    # 释放资源
    video.release()


def show3Dpose(vals, ax, fix_z):
    ax.view_init(elev=15., azim=70)

    colors = [(138 / 255, 201 / 255, 38 / 255),
              (255 / 255, 202 / 255, 58 / 255),
              (25 / 255, 130 / 255, 196 / 255)]

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = [3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1]

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=3, color=colors[LR[i] - 1])

    RADIUS = 0.72

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    if fix_z:
        left_z = max(0.0, -RADIUS + zroot)
        right_z = RADIUS + zroot
        # ax.set_zlim3d([left_z, right_z])
        ax.set_zlim3d([0, 1.5])
    else:
        ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])

    ax.set_aspect('equal')  # works fine in matplotlib==2.2.2 or 3.7.1

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)


def tensorVis(res, name):
    res = res.squeeze(0)
    frame = res.shape[0]
    numpy_array = res.detach().cpu().numpy()
    rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    numpy_array = camera_to_world(numpy_array, R=rot, t=0)
    output_dir_3D = 'pose3D/' + name + '/'
    for jj in range(frame):
        ## 3D
        fig = plt.figure(figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05)
        ax = plt.subplot(gs[0], projection='3d')

        numpy_array[jj, :, 2] -= np.min(numpy_array[jj, :, 2])
        show3Dpose(numpy_array[jj], ax, False)

        os.makedirs(output_dir_3D, exist_ok=True)
        plt.savefig(output_dir_3D + str(('%04d' % jj)) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
        plt.close()
    video_name = output_dir_3D + '/' + name + '.mp4'
    img2mp4(output_dir_3D, video_name)


if __name__ == '__main__':
    rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    npz_file_path = '../dataset/video/' + 'breakdance' + '/' + 'abe' + '/output_3D/output_keypoints_3d.npz'
    # 使用np.load()函数加载npz文件
    with np.load(npz_file_path) as data:
        # 列出文件中的所有数组名称
        arr_0 = data['reconstruction']
        for jj in range(30):
            ## 3D
            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05)
            ax = plt.subplot(gs[0], projection='3d')

            arr_0[jj, :, 2] -= np.min(arr_0[jj, :, 2])
            show3Dpose(arr_0[jj], ax, False)

            output_dir_3D = 'pose3D/'
            os.makedirs(output_dir_3D, exist_ok=True)
            plt.savefig(output_dir_3D + str(('%04d' % jj)) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
            plt.close()
