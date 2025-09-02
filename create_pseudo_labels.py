import os
import argparse
from PIL import Image

import cv2
import numpy as np
import torch
import torchvision
from segment_anything_hq import (
    SamAutomaticMaskGenerator,
    sam_model_registry,
)
from sklearn.cluster import KMeans

from composition_map_creation.dataset import MVTecLocoDataset
from composition_map_creation.modules import DinoFeaturizer


def transform_to_mask_array(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    mask_array = np.zeros(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            len(sorted_anns),
        )
    )

    i = 0
    for ann in sorted_anns:
        if ann["area"] < 50:
            mask_array = mask_array[:, :, : i - 1]
            break
        m = ann["segmentation"]
        mask_array[m, :] = 0
        mask_array[:, :, i] = m
        i = i + 1
    return mask_array

def get_features(img, bg, backbone):
    img = img.unsqueeze(0)
    with torch.no_grad():
        features, _ = backbone(img)
    features = torchvision.transforms.Resize((256, 256))(features)
    bg = bg.unsqueeze(1).expand(-1, 768, -1, -1)
    features = features * bg
    features = features.squeeze().cpu().numpy().transpose((1, 2, 0))
    return features

def create_refined_masks(i, mask, bg, save_path, palette):
    # mask[bg] = 0
    mask = np.where(bg == 1, 0, mask)
    mask = mask.astype(np.uint8)
    pi = Image.fromarray(mask, 'P')
    pi.putpalette(palette)
    pi.show()
    pi.save(f"{save_path}/{i}_refined_seg.png")

def create_noisy_masks(dataset, c, backbone):
    all_masks = np.zeros((len(dataset), 256, 256))
    for i in range(len(dataset)):
        print(i)
        img = cv2.imread(dataset.image_paths[i])
        img = cv2.resize(img, (256, 256))

        data = dataset.__getitem__(i)
        feat = (
            get_features(data["image"].cuda(), data["bg"].cuda(), backbone)
            .reshape(256 * 256, 768)
            .astype(np.float64)
        )
        mask = c.predict(feat)
        mask = mask.reshape(256, 256)
        mask = mask + 1

        bg = 1 - data["bg"].squeeze().numpy()
        mask[np.where(bg == 1)] = 0
        all_masks[i, :, :] = mask

    return all_masks

def setup_sam(model_path):
    sam_checkpoint = f"{model_path}sam_hq_vit_h.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=64,
        pred_iou_thresh=0.8,
        min_mask_region_area=50,
        crop_n_layers=0,
    )
    return mask_generator


def refine_masks_with_sam(dataset, masks_dino, save_path, model_path):
    palette = [0, 0, 0, 204, 241, 227, 112, 142, 18, 254, 8, 23, 207, 149, 84, 202, 24, 214,
               230, 192, 37, 241, 80, 68, 74, 127, 0, 2, 81, 216, 24, 240, 129, 20, 215, 125, 161, 31, 204,
               254, 52, 116, 117, 198, 203, 4, 41, 68, 127, 252, 61, 21, 3, 142, 40, 10, 159, 241, 61, 36,
               14, 175, 77, 144, 61, 115, 131, 79, 97, 109, 177, 163, 58, 198, 140, 17, 235, 168, 47, 128, 91,
               238, 103, 45, 124, 35, 228, 101, 48, 232, 74, 124, 114, 78, 49, 30, 35, 167, 27, 137, 231, 47,
               235, 32, 39, 56, 112, 32, 62, 173, 79, 86, 44, 201, 77, 47, 217, 246, 223, 57, ]
    # Pad with zeroes to 768 values, i.e. 256 RGB colours
    palette = palette + [0]*(768-len(palette))

    refined_masks = np.zeros((len(dataset), 256, 256))
    mask_generator = setup_sam(model_path)
    
    for i in range(len(dataset)):
        print(i)

        mask_dino = masks_dino[i, :, :]
        data = dataset.__getitem__(i)
        img = cv2.imread(dataset.image_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        

        with torch.no_grad():
            masks = mask_generator.generate(img)
            masks = transform_to_mask_array(
                masks
            )  

        bg = data["bg"].squeeze().unsqueeze(-1).cpu().numpy()
        masks = np.where(bg == 0, 0, masks)
        # idx = np.where(bg == 0)  # Remove bg from sam masks

        # for x, y in zip(idx[0], idx[1]):
        #     masks[x, y, :] = 0
        non_zero_arrays = np.any(masks, axis=(0, 1))
        zero_arrays_indices = np.where(~non_zero_arrays)
        masks = np.delete(masks, zero_arrays_indices, axis=-1)
        end_array = np.zeros((256, 256))
        for j in range(masks.shape[-1]):
            curr_mask = masks[:, :, j]
            idx = np.where(curr_mask == 1)
            masked_array = np.ma.masked_array(mask_dino, curr_mask == 0)
            flattened_array = masked_array.compressed()
            counts = np.bincount(flattened_array.astype(np.int64))
            most_common_value = np.argmax(counts)
            end_array[curr_mask == 1] = most_common_value
        bg = 1 - data["bg"].squeeze().numpy()
        refined_masks[i, :, :] = end_array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{save_path}/{i}_gt.png", img)
        create_refined_masks(i, end_array, bg, save_path, palette)
    return refined_masks


def get_masks(args):
    data_path = args.data_path + "/" + args.category + "/"
    mask_path = args.mask_path + "/" + args.category + "/"
    save_path = (
        args.save_path
        + "/"
        + args.category
        + "/train/good/"
    )
    model_path = args.pretrained_model_path

    os.makedirs(save_path, exist_ok=True)

    dataset = dataset = MVTecLocoDataset(
        root_dir=data_path, bg_dir=mask_path, category="/train/good", resize_shape=512
    )

    backbone = DinoFeaturizer().cuda()
    sample_size = 256 * 256 // 500 # For memory reasons
    # sample_size = 256 * 256 // 10#// 500 # For memory reasons
    all_feats = np.zeros((len(dataset) * sample_size, 768))

    for i in range(len(dataset)):
        data = dataset.__getitem__(i)

        feat = get_features(data["image"].cuda(), data["bg"].cuda(), backbone).reshape(
            (-1, 768)
        )
        background = data["bg"].squeeze().numpy().reshape(-1)
        idx = np.argwhere(background == 1)
        random_index = idx[np.random.choice(len(idx), sample_size, replace=False)]
        all_feats[i * sample_size : (i + 1) * sample_size, :] = feat[
            random_index, :
        ].squeeze(1)

    kmean = KMeans(init="k-means++", n_clusters=args.n_clusters)
    c = kmean.fit(all_feats)
    all_masks = create_noisy_masks(dataset, c, backbone)
    refine_masks_with_sam(dataset, all_masks, save_path, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", default="./data/mvtec_loco"
    )
    parser.add_argument(
        "--mask-path", default="./data/mvtec_loco_masks"
    )
    parser.add_argument(
        "--save-path",
        default="./data/mvtec_loco_noisy_composition_maps",
    )
    parser.add_argument(
        "--pretrained-model-path",
        default="./pretrained_models/",
    )
    parser.add_argument("--category", default="screw_bag")
    parser.add_argument("--n-clusters", type=int, default=5)

    args = parser.parse_args()

    get_masks(args)
