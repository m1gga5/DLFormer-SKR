import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MixDataset(Dataset):
    def __init__(self, inputs_dir):
        self.inputs_dir = inputs_dir
        self.inputs, self.targets = self.read_inputs()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        targets = self.targets[idx]
        return inputs, targets

    def read_inputs(self):
        tensors = []
        targets = []
        entries = os.listdir(self.inputs_dir)
        # 筛选出所有文件夹
        names = [entry for entry in entries if os.path.isdir(os.path.join(self.inputs_dir, entry))]
        for name in names:
            npz_path = os.path.join(self.inputs_dir, name, 'output_3D/output_keypoints_3d.npz')
            with np.load(npz_path) as data:
                # 列出文件中的所有数组名称
                np_arr = data['reconstruction']
                tensor = torch.from_numpy(np_arr).float()
                tensors.append(tensor)
                targets.append(name)
        return tensors, targets


def load_target(npz_path):
    with np.load(npz_path) as data:
        # 列出文件中的所有数组名称
        np_arr = data['reconstruction']
        tensor = torch.from_numpy(np_arr).float()
        tensor = torch.unsqueeze(tensor, 0)
        return tensor
