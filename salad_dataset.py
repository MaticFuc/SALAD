from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torch
import math
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
import numpy as np
from copy import deepcopy
from scipy.ndimage import label, binary_dilation
import cv2
import os
from PIL import Image
import torch.functional as F


def index_loader(path):
    img = Image.open(path)
    return img

default_transform = transforms.Compose([
    transforms.ToTensor(),
])

mask_aug = transforms.Compose([
    transforms.RandomRotation(90),
    # transforms.RandomAffine(0,(0.25,0.25))
])

palette = [0, 0, 0, 204, 241, 227, 112, 142, 18, 254, 8, 23, 207, 149, 84, 202, 24, 214,
                    230, 192, 37, 241, 80, 68, 74, 127, 0, 2, 81, 216, 24, 240, 129, 20, 215, 125, 161, 31, 204,
                    254, 52, 116, 117, 198, 203, 4, 41, 68, 127, 252, 61, 21, 3, 142, 40, 10, 159, 241, 61, 36,
                    14, 175, 77, 144, 61, 115, 131, 79, 97, 109, 177, 163, 58, 198, 140, 17, 235, 168, 47, 128, 91,
                    238, 103, 45, 124, 35, 228, 101, 48, 232, 74, 124, 114, 78, 49, 30, 35, 167, 27, 137, 231, 47,
                    235, 32, 39, 56, 112, 32, 62, 173, 79, 86, 44, 201, 77, 47, 217, 246, 223, 57, ]
palette = palette + [0]*(768-len(palette))

def rand_perlin_2d(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])
            ),
            dim=-1,
        )
        % 1
    )
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = (
        lambda slice1, slice2: gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
        .repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
    )
    dot = lambda grad, shift: (
        torch.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            dim=-1,
        )
        * grad[: shape[0], : shape[1]]
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * torch.lerp(
        torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1]
    )

def get_perlin_noise(x0):
    perlin_scale = 6
    min_perlin_scale = 0
    perlin_scalex = 2 ** (
        torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
    )
    perlin_scaley = 2 ** (
        torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
    )
    noiseMap = rand_perlin_2d(
        (x0.shape[1], x0.shape[2]), (perlin_scalex, perlin_scaley)
    )
    rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
    noiseMap = rot(image=np.array(noiseMap))
    noiseMap = noiseMap > 0.5
    noiseMap = noiseMap.astype(int).astype(float)
    return torch.FloatTensor(noiseMap)

def create_draem_anomalies(anom_seg_mask, diff_seg_mask):
    mask = get_perlin_noise(anom_seg_mask)
    C = anom_seg_mask.shape[0]
    ci = torch.randint(C, (1,)).item()

    recon_mask = torch.logical_and(mask, anom_seg_mask[ci, :, :]).type(
        torch.FloatTensor
    )
    mask = torch.logical_and(mask, 1-anom_seg_mask[ci, :, :]).type(
        torch.FloatTensor
    )
    idx = torch.where(anom_seg_mask == mask.expand(C, -1, -1))
    anom_seg_mask[idx] = 0
    anom_seg_mask[ci, :, :] = torch.maximum(mask, anom_seg_mask[ci, :, :])
    mask = mask.squeeze(0)
    # recon_mask = torch.zeros_like(mask)
    return mask, recon_mask, anom_seg_mask

def get_connected_component_at_x_y(mask, x, y, channel):
    H, W = mask.shape[1], mask.shape[2]
    visited = np.zeros((H, W), dtype=bool)
    out_mask = np.zeros((H,W), dtype=bool)

    # Check if starting point is part of a component
    if mask[channel, x, y] == 0:
        return out_mask  # No component if starting point is 0

    # Stack-based DFS to find connected component
    stack = [(x, y)]
    while stack:
        i, j = stack.pop()
        if visited[i, j] or mask[channel, i, j] == 0:
            continue
        visited[i, j] = True
        out_mask[i, j] = True

        # Check 4-connected neighbors (up, down, left, right)
        for di, dj in [(-1, 0), (-1, 1), (-1, -1), (1, 0), (1, 1), (1, -1), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not visited[ni, nj]:
                stack.append((ni, nj))

    return out_mask

def get_connected_components(mask):
    # Assuming mask is of size (C, H, W)
    components_list = []
    components_ci = []

    for c in range(mask.shape[0]):  # Iterate over channels
        # Convert the mask to numpy for processing
        mask_numpy = mask[c].cpu().numpy()  # Assuming mask is on GPU, move to CPU
        
        # Label connected components
        labeled_mask, num_features = label(mask_numpy)
        
        
        # Collect each component into the list
        for i in range(1, num_features + 1):  # 0 is background
            component = labeled_mask == i
            if component.sum() > 200:
                components_list.append(component)
                components_ci.append(c)
    
    return components_list, components_ci

def change_label_same_img(seg_mask, diff_seg_mask):
    x, y = torch.randint(0, seg_mask.shape[-1], (2,))
    mask = torch.zeros((256, 256))
    recon_mask = torch.zeros((256, 256))
    n_new_labels = torch.randint(1, 2, (1,)).item()
    for _ in range(n_new_labels):

        component_list, ci_list = get_connected_components(seg_mask)
        i = torch.randint(0, len(component_list), (1,)).item()
        connected_component, ci = component_list[i], ci_list[i]

        mask[connected_component] = 1
        dilated_mask = binary_dilation(mask.numpy(), structure=[[1,1,1], [1,1,1], [1,1,1]])

        dilated_mask = torch.tensor(dilated_mask, dtype=torch.bool)
        dilated_mask[mask.numpy() == 1] = 0 

        neighbor_channels = torch.where((seg_mask[:, dilated_mask]).sum(dim=(1)) > 0)[0]
        ci = neighbor_channels[torch.randint(0,len(neighbor_channels), (1,)).item()]
        
        recon_mask[connected_component] = 1

        seg_mask[:, connected_component] = 0
        seg_mask[ci, connected_component] = 1


        idx = np.argwhere(connected_component)[0]
        x, y = idx[0], idx[1]
        connected_component = get_connected_component_at_x_y(seg_mask, x, y, ci)
        mask[connected_component] = 1
    
    return mask, recon_mask, seg_mask

def copy_label_diff_img(seg_mask, diff_seg_mask):
    mask = torch.zeros((256, 256))
    n_new_labels = 1
    for _ in range(n_new_labels):
    
        component_list, ci_list = get_connected_components(diff_seg_mask)
        i = torch.randint(0, len(component_list), (1,)).item()
        connected_component, ci = component_list[i], ci_list[i]
        mask[connected_component] = 1
        mask = mask_aug(mask.unsqueeze(0)).squeeze()
        mask = mask.to(torch.bool).unsqueeze(0)
        mask = mask.repeat((seg_mask.shape[0],1,1))

        seg_mask[mask] = 0
        mask = mask[0,:,:]
        seg_mask[ci, mask] = 1
        mask = mask.to(torch.float)
        recon_mask = seg_mask[ci,:,:]
        mask = recon_mask
    return mask, recon_mask, seg_mask

def copy_and_change(seg_mask, diff_seg_mask):
    mask_1, recon_mask_1, seg_mask = change_label_same_img(seg_mask, diff_seg_mask)
    mask_2, recon_mask_2, seg_mask = copy_label_diff_img(seg_mask, diff_seg_mask)
    mask = torch.maximum(mask_1, mask_2)
    recon_mask = torch.maximum(recon_mask_1, recon_mask_2) 
    return mask, recon_mask, seg_mask
    

def read_mask(img, num_cls):
    # img_tensor = torch.from_numpy(np.array(img, dtype=np.uint8)).long()
    
    # # Apply one-hot encoding and reshape to [num_cls, H, W]
    # onehot = F.one_hot(img_tensor, num_classes=num_cls).permute(2, 0, 1).float()
    
    # return onehot

    img = torch.LongTensor(np.array(img, dtype=np.uint8))
    img = img.unsqueeze(0)
    
    # print(img.shape, torch.unique(img))

    
    onehot_img = torch.zeros_like(img).repeat(num_cls, 1, 1)
    # print(img.shape, onehot_img.shape)
    onehot_img = onehot_img.scatter(0, img.long(), 1)
    return onehot_img.float()


class ImageFolderWithoutTargetWithSeg(Dataset):

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.train = False
        self.mask_fns = [create_draem_anomalies, change_label_same_img, copy_label_diff_img, copy_and_change] 
        

    def __getitem__(self, index):
        img = self.dataset1.__getitem__(index)
        seg = self.dataset2.__getitem__(index)

        random_index = torch.randint(0,len(self.dataset2),(1,)).item()
        diff_seg = self.dataset2.__getitem__(random_index)
        
        mask, recon_mask, anom_seg = torch.zeros((256,256)), torch.ones((256,256)), deepcopy(seg)
        if self.train and torch.rand((1,)).item() > 0.5:
            idx = torch.randint(0,len(self.mask_fns),(1,)).item()
            mask, recon_mask, anom_seg = self.mask_fns[idx](anom_seg, diff_seg)
            recon_mask = 1 - recon_mask

        return img, seg, anom_seg, mask, recon_mask

    def __len__(self):
        return len(self.dataset1)
    
    
class ImageFolderWithPathWithSeg(Dataset):

    def __init__(self, dataset1, dataset2, puad=False, num_cls=6):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.puad = puad
        self.num_cls = num_cls


    def __getitem__(self, index):
        img, target, path = self.dataset1.__getitem__(index)
        mask_path = path.replace("test", "ground_truth").replace(".png","/000.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256,256))
        else:
            mask = torch.zeros((256, 256))
        mask = torch.FloatTensor(mask / 255.0)       
        seg = self.dataset2.__getitem__(index)
        return img, seg, mask, path
    
    def __len__(self):
        return len(self.dataset1)

class ImageFolderWithoutTarget(ImageFolder):

    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.seg:
            sample = Image.open(path)
            sample = read_mask(sample, 6)
        else:
            sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        if self.seg:
            sample = Image.open(path)
        return sample, target, path
def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
