import torch
import torch.nn.functional as F
import numpy as np


def lucas_kanade_optical_flow(prev_pts, next_pts):
    flow = next_pts - prev_pts

    return flow


def lucas_kanade_optical_flow2(first, pts):
    # 确保输入张量是浮点类型，以便进行计算
    pts = pts.float()

    # 初始化输出张量，填充为0
    B, T, V, C = pts.shape
    flow = torch.zeros((B, T, V, C), dtype=torch.float32, device=pts.device)

    # 计算每个点相对于前一个点的偏移
    for b in range(B):  # 遍历批次
        for t in range(0, T):  # 从第0帧开始
            for v in range(V):  # 遍历所有点
                # 计算偏移量，即当前点减去前一个点
                if t == 0:
                    flow[b, t, v] = pts[b, t, v] - first[b, t, v]
                else:
                    flow[b, t, v] = pts[b, t, v] - pts[b, t - 1, v]

    return flow

