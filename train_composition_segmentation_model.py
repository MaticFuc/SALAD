import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from composition_map_creation.dataset import MVTecLocoTestValDataset
from composition_map_creation.pseudo_label_dataset import PseudoLabelDataset
from composition_map_creation.unet import UNet


IMAGENET_INV_MEAN = [-0.485, -0.456, -0.406]
IMAGENET_ZEROS = [0.0, 0.0, 0.0]
IMAGENET_ONES = [1.0, 1.0, 1.0]
IMAGENET_INV_STD = [1 / 0.229, 1 / 0.224, 1 / 0.225]

inv_normalize = transforms.Compose(
    [
        transforms.Normalize(mean=IMAGENET_ZEROS, std=IMAGENET_INV_STD),
        transforms.Normalize(mean=IMAGENET_INV_MEAN, std=IMAGENET_ONES),
    ]
)



def save_img(i, savepath, mask, palette):

    mask = mask.detach().cpu().numpy().astype(np.uint8)
    pi = Image.fromarray(mask, 'P')
    pi.putpalette(palette)
    pi.show()
    pi.save(f"{savepath}{i:03d}.png")


def get_model():
    parameters = {
        "n_channels": 64,
        "image_channels": 3,
        "output_channels": 6,
        "is_attn": [False, False, False, False],
        "ch_mults": [1, 2, 2, 2],
        "n_blocks": 1,
        "append_positional_embedding_start": False,
        "add_positional_embedding_start": False,
    }
    model = UNet(parameters)
    return model


def test(testset, model, save_path, test_partition):
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    palette = [0, 0, 0, 204, 241, 227, 112, 142, 18, 254, 8, 23, 207, 149, 84, 202, 24, 214,
               230, 192, 37, 241, 80, 68, 74, 127, 0, 2, 81, 216, 24, 240, 129, 20, 215, 125, 161, 31, 204,
               254, 52, 116, 117, 198, 203, 4, 41, 68, 127, 252, 61, 21, 3, 142, 40, 10, 159, 241, 61, 36,
               14, 175, 77, 144, 61, 115, 131, 79, 97, 109, 177, 163, 58, 198, 140, 17, 235, 168, 47, 128, 91,
               238, 103, 45, 124, 35, 228, 101, 48, 232, 74, 124, 114, 78, 49, 30, 35, 167, 27, 137, 231, 47,
               235, 32, 39, 56, 112, 32, 62, 173, 79, 86, 44, 201, 77, 47, 217, 246, 223, 57, ]
    # Pad with zeroes to 768 values, i.e. 256 RGB colours
    palette = palette + [0]*(768-len(palette))
    with tqdm(
        total=len(testset),
        desc=f"TEST composition map segmentation ({test_partition})",
        unit="img",
    ) as prog_bar:
        epoch_save_path = save_path
        os.makedirs(epoch_save_path, exist_ok=True)
        for i, data in enumerate(testloader):
            img = data["image"].cuda()
            with torch.no_grad():
                predicted_mask = model(img)
            predicted_mask = predicted_mask.softmax(dim=1).argmax(dim=1)
            save_img(
                i, epoch_save_path, predicted_mask.squeeze(0), palette
            )
            prog_bar.update(1)


def train(args):
    save_path = args.log_path + "/" + args.category
    os.makedirs(save_path, exist_ok=True)
    path = (
        args.data_path
        + "/"
        + args.category
        + f"/{args.n_clusters}_clusters"
        + "/train/good/"
    )
    model = get_model().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        [100],
        # [int(args.epochs * 0.80), int(args.epochs * 0.90)],
        gamma=0.2,
        last_epoch=-1,
    )
    dataset = PseudoLabelDataset(
        {"path": path, "size": 256}
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    img_path = args.img_path + "/" + args.category
    # print(img_path)
    testset = dict()
    t = os.listdir(f"{img_path}/test/")
    for ti in t:
        testset[ti] = MVTecLocoTestValDataset(
            root_dir=img_path, category=f"/test/{ti}", resize_shape=256
        )

    valset = MVTecLocoTestValDataset(
        root_dir=img_path, category=f"/validation/good", resize_shape=256
    )

    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        cnt = 0
        with tqdm(
            total=len(dataset),
            desc=f"TRAIN Composition map segmentation - Epoch {epoch}: ",
            unit="img",
        ) as prog_bar:
            for i, data in enumerate(dataloader):
                img = data["image"].cuda()
                mask = data["mask"].cuda().type(torch.cuda.LongTensor)
                optimizer.zero_grad()
                predicted_mask = model(img).softmax(dim=1)
                loss = loss_fn(predicted_mask, mask).mean()
                total_loss += loss.item()
                cnt += img.shape[0]
                loss.backward()
                optimizer.step()
                prog_bar.set_postfix_str(f"loss: {total_loss / cnt}")
                prog_bar.update(img.shape[0])

    model.eval()
    test(
        dataset,
        model,
        save_path + f"/train/good/",
        "Train set",
    )
    for ti in testset.keys():
        test(
            testset[ti],
            model,
            save_path + f"/test/{ti}/",
            f"Test set - {ti}",
        )
    test(
        valset,
        model,
        save_path + f"/validation/good/",
        f"Validation set - {ti}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", default="./data/mvtec_loco")
    parser.add_argument(
        "--data-path",
        default=".data/mvtec_loco_noisy_composition_maps",
    )
    parser.add_argument(
        "--log-path", default="./data/mvtec_loco_composition_maps"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--category", default="pushpins")
    parser.add_argument("--n-clusters", type=int, default=5)

    args = parser.parse_args()
    train(args)
