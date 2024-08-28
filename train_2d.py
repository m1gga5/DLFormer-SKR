import os
import random

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.BSTformer import BSTformer
from utils.load_json import ToTensor, PoseDataset, SeqDataset, load_pose, load_target, MixDataset

motion = ['bboy_uprock_start', 'breakdance', 'chicken_dance', 'dancing_twerk', 'dodging_right', 'samba_dancing',
          'goalkeeper_catch', 'hiphop_dancing', 'injured_walk',
          'jogging', 'joyful_jump', 'jump', 'rumba_dancing', 'salsa_dancing', 'silly_dancing', 'snatch', 'roar',
          'swing_dancing', 'twist_dance']
names = ['abe', 'brian', 'drake', 'elizabeth', 'james', 'leonard', 'lewis', 'maynard', 'megan', 'remy', 'shannon',
         'sophie']


def train_epoch(num_epochs, model_pos, train_loader, optimizer, criterion, scheduler):
    model_pos = model_pos.to(device)
    # criterion = criterion.to(device)
    min_loss = 100000
    # file_path = "training_losses.txt"
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        avg_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}', unit='batches') as pbar:
            for idx, (batch_input, batch_gt) in enumerate(train_loader):
                batch_input = batch_input.to(device)
                targets = []
                standards = []
                for name in batch_gt:
                    random_motion = random.choice(motion)
                    random_name = random.choice(names)
                    standard = load_target(os.path.join('datasets/train/json/seq', random_motion), random_name)
                    target = load_target(os.path.join('datasets/train/json/seq', random_motion), name)
                    standards.append(standard)
                    targets.append(target)
                standards = torch.cat(standards, 0)
                targets = torch.cat(targets, 0)
                standards = standards.to(device)
                targets = targets.to(device)
                outputs = model_pos(standards, batch_input)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()  # 清除之前的梯度
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                epoch_loss += loss.item()
                avg_loss = epoch_loss / (idx + 1)
                pbar.set_postfix(loss=f'{avg_loss:.3f}', lr=f'{scheduler.get_last_lr()[0]:.10f}',
                                 min_loss=f'{min_loss:.3f}')
                pbar.update(1)
            if avg_loss < min_loss:
                min_loss = avg_loss
                torch.save(model_pos.state_dict(), 'best.pth')
                # with open(file_path, "a") as file:
                #     file.write(
                #         f"Epoch {epoch + 1}:loss={avg_loss:.3f},lr={scheduler.get_last_lr()[0]:.5f}, min_loss={min_loss:.3f}\n")
            scheduler.step()


def loss_2d_weighted(predicted, target):
    assert predicted.shape == target.shape
    predicted_2d = predicted[:, :, :, :2]
    target_2d = target[:, :, :, :2]
    diff = (predicted_2d - target_2d)
    return torch.mean(torch.norm(diff, dim=-1))


if __name__ == "__main__":
    model_backbone = BSTformer(dim_in=2, dim_out=2, num_joints=14, depth=5)
    num_parameters = sum(p.numel() for p in model_backbone.parameters())
    print(f"Number of parameters: {num_parameters}")
    transform = ToTensor()

    pose_dataset = MixDataset(inputs_dir='datasets/train/json/seq/stands',
                              transform=transform)

    data_loader = DataLoader(pose_dataset, batch_size=4, shuffle=True)

    optimizer = optim.Adam(model_backbone.parameters(), lr=0.0002)
    criterion = loss_2d_weighted
    num_epochs = 600
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_backbone.load_state_dict(torch.load('best.pth', map_location=device))

    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    people_names = []
    model_backbone.train()
    train_epoch(num_epochs, model_backbone, data_loader, optimizer, criterion, scheduler)
