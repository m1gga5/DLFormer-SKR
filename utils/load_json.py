import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset


class ToTensor(object):
    def __call__(self, sample):
        # 检查 sample 是否为 NumPy 数组
        if isinstance(sample, np.ndarray):
            # 如果是 NumPy 数组，使用 torch.from_numpy() 转换
            tensor = torch.from_numpy(sample).float()
        elif isinstance(sample, torch.Tensor):
            # 如果已经是 Tensor，直接转换为 float32 类型
            tensor = sample.float()
        else:
            raise TypeError("sample must be either a NumPy array or a Torch Tensor")
        return tensor


class PoseDataset(Dataset):
    def __init__(self, inputs_dir, targets_dir, transform=None):
        self.inputs_dir = inputs_dir
        self.targets_dir = targets_dir
        self.transform = transform
        self.inputs = self.read_inputs()
        self.targets = self.read_targets()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        targets = self.targets[idx]
        if self.transform:
            inputs = self.transform(inputs)
            targets = self.transform(targets)
        return inputs, targets

    def read_inputs(self):
        tensors = []
        for file in os.listdir(self.inputs_dir):
            file_path = os.path.join(self.inputs_dir, file)

            if file.endswith('.json'):
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)

                keypoints_2d = data['people'][0]['pose_keypoints_2d'][:28]
                keypoints_array = np.array(keypoints_2d, dtype=np.float32).reshape((14, 2))
                keypoints_tensor = torch.from_numpy(keypoints_array).float()
                # 添加通道维度到第0维
                keypoints_tensor = np.expand_dims(keypoints_tensor, axis=0)
                tensors.append(keypoints_tensor)
        return tensors

    def read_targets(self):
        tensors = []
        for file in os.listdir(self.targets_dir):
            file_path = os.path.join(self.targets_dir, file)

            if file.endswith('.json'):
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)

                keypoints_2d = data['people'][0]['pose_keypoints_2d'][:28]
                keypoints_array = np.array(keypoints_2d, dtype=np.float32).reshape((14, 2))
                keypoints_tensor = torch.from_numpy(keypoints_array).float()
                # 添加通道维度到第0维
                keypoints_tensor = np.expand_dims(keypoints_tensor, axis=0)
                tensors.append(keypoints_tensor)
        return tensors


class SeqDataset(Dataset):
    def __init__(self, inputs_dir, targets_dir, transform=None):
        self.inputs_dir = inputs_dir
        self.targets_dir = targets_dir
        self.transform = transform
        self.inputs = self.read_inputs()
        self.targets = self.inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        targets = self.targets[idx]
        if self.transform:
            inputs = self.transform(inputs)
            targets = self.transform(targets)
        return inputs, targets

    def read_inputs(self):
        batches = []
        for subdir in os.listdir(self.inputs_dir):
            subdir_path = os.path.join(self.inputs_dir, subdir)
            if os.path.isdir(subdir_path):
                tensors = []
                json_files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.json')])
                for file in json_files:
                    file_path = os.path.join(subdir_path, file)
                    if file.endswith('.json'):
                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)
                        keypoints_2d = data['people'][0]['pose_keypoints_2d'][:28]
                        keypoints_array = np.array(keypoints_2d, dtype=np.float32).reshape((14, 2))
                        keypoints_tensor = torch.from_numpy(keypoints_array).float()
                        keypoints_tensor = torch.unsqueeze(keypoints_tensor, 0)
                        tensors.append(keypoints_tensor)
                deeps = torch.cat(tensors, 0)
                batches.append(deeps)
        return batches

    def read_targets(self):
        batches = []
        for subdir in os.listdir(self.targets_dir):
            subdir_path = os.path.join(self.targets_dir, subdir)
            if os.path.isdir(subdir_path):
                tensors = []
                json_files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.json')])
                for file in json_files:
                    file_path = os.path.join(subdir_path, file)
                    if file.endswith('.json'):
                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)
                        keypoints_2d = data['people'][0]['pose_keypoints_2d'][:28]
                        keypoints_array = np.array(keypoints_2d, dtype=np.float32).reshape((14, 2))
                        keypoints_tensor = torch.from_numpy(keypoints_array).float()
                        keypoints_tensor = torch.unsqueeze(keypoints_tensor, 0)
                        tensors.append(keypoints_tensor)
                deeps = torch.cat(tensors, 0)
                batches.append(deeps)
        return batches


class MixDataset(Dataset):
    def __init__(self, inputs_dir, transform=None):
        self.inputs_dir = inputs_dir
        self.transform = transform
        self.inputs, self.targets = self.read_inputs()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        targets = self.targets[idx]
        if self.transform:
            inputs = self.transform(inputs)
        return inputs, targets

    def read_inputs(self):
        tensors = []
        targets = []
        for file in os.listdir(self.inputs_dir):
            file_path = os.path.join(self.inputs_dir, file)

            if file.endswith('.json'):
                target = file.split('_')[0]
                targets.append(target)
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)

                keypoints_2d = data['people'][0]['pose_keypoints_2d'][:28]
                keypoints_array = np.array(keypoints_2d, dtype=np.float32).reshape((14, 2))
                keypoints_tensor = torch.from_numpy(keypoints_array).float()
                # 添加通道维度到第0维
                keypoints_tensor = np.expand_dims(keypoints_tensor, axis=0)
                tensors.append(keypoints_tensor)
        return tensors, targets



def load_pose(folder_path, name):
    # 确保文件夹路径以斜杠结束
    # if not folder_path.endswith('\\'):
    #     folder_path += '\\'

    # 遍历文件夹内的所有文件和文件夹
    for filename in os.listdir(folder_path):
        # 检查文件是否以提供的name开头，并且是.json文件
        if filename.startswith(name) and filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            keypoints_2d = data['people'][0]['pose_keypoints_2d'][:28]
            keypoints_array = np.array(keypoints_2d, dtype=np.float32).reshape((14, 2))
            keypoints_tensor = torch.from_numpy(keypoints_array).float()
            keypoints_tensor = torch.unsqueeze(keypoints_tensor, 0)
            pose = torch.unsqueeze(keypoints_tensor, 0)
            return pose
    # 如果没有找到文件，返回None
    return None


def load_target(folder_path, name):
    # 确保文件夹路径以斜杠结束
    # if not folder_path.endswith('\\'):
    #     folder_path += '\\'

    whole_path = os.path.join(folder_path, name)

    if os.path.isdir(whole_path):
        tensors = []
        json_files = sorted([f for f in os.listdir(whole_path) if f.endswith('.json')])
        for file in json_files:
            file_path = os.path.join(whole_path, file)
            if file.endswith('.json'):
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                keypoints_2d = data['people'][0]['pose_keypoints_2d'][:28]
                keypoints_array = np.array(keypoints_2d, dtype=np.float32).reshape((14, 2))
                keypoints_tensor = torch.from_numpy(keypoints_array).float()
                keypoints_tensor = torch.unsqueeze(keypoints_tensor, 0)
                tensors.append(keypoints_tensor)
        deeps = torch.cat(tensors, 0)
        batch = torch.unsqueeze(deeps, 0)
        return batch


def load_batch(folder_path, name):
    # 确保文件夹路径以斜杠结束
    # if not folder_path.endswith('\\'):
    #     folder_path += '\\'

    whole_path = os.path.join(folder_path, name)

    if os.path.isdir(whole_path):
        tensors = []
        json_files = sorted([f for f in os.listdir(whole_path) if f.endswith('.json')])
        for file in json_files:
            file_path = os.path.join(whole_path, file)
            if file.endswith('.json'):
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                keypoints_2d = data['people'][0]['pose_keypoints_2d'][:28]
                keypoints_array = np.array(keypoints_2d, dtype=np.float32).reshape((14, 2))
                keypoints_tensor = torch.from_numpy(keypoints_array).float()
                keypoints_tensor = torch.unsqueeze(keypoints_tensor, 0)
                tensors.append(keypoints_tensor)
        deeps = torch.cat(tensors, 0)
        batch = torch.unsqueeze(deeps, 1)
        return batch