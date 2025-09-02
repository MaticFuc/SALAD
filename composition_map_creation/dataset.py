import glob
import random

import PIL
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

resize_size = 256


def choose_random_aug_image(image):
    aug_index = random.choice([1, 2, 3])
    coefficient = random.uniform(0.8, 1.2)
    if aug_index == 1:
        img_aug = transforms.functional.adjust_brightness(image, coefficient)
    elif aug_index == 2:
        img_aug = transforms.functional.adjust_contrast(image, coefficient)
    elif aug_index == 3:
        img_aug = transforms.functional.adjust_saturation(image, coefficient)
    return img_aug


class MVTecLocoDataset(Dataset):
    def __init__(self, root_dir, bg_dir, category, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.is_good = ("good" in category) or ("ok" in category)
        self.resize_shape = resize_shape
        self.category = category
        img_path = root_dir + f"{self.category}/*"
        mask_path = img_path.replace("test", "ground_truth") + "/000.png"
        img_path = img_path + ".png"
        bg_path = bg_dir + f"{self.category}/*.png"
        self.image_paths = list(sorted(glob.glob(img_path)))
        self.mask_paths = list(sorted(glob.glob(mask_path)))
        self.bg_paths = list(sorted(glob.glob(bg_path)))
        self.foreground_masks = None
        self.transform_img = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        imageo = PIL.Image.open(self.image_paths[idx]).convert("RGB")
        bg = PIL.Image.open(self.bg_paths[idx]).convert("L")
        totensor = transforms.ToTensor()
        
        if self.is_good:
            mask = torch.zeros((self.resize_shape, self.resize_shape))
        else:
            mask = PIL.Image.open(self.mask_paths[idx]).convert("L")
            mask = self.transform_mask(mask)
        image = self.transform_img(imageo)
        bg = totensor(bg)
        Image = {
            "image": image,
            "mask": mask,
            "bg": bg,
        }
        return Image


class MVTecLocoLogicalDataset(Dataset):
    def __init__(self, root_dir, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape

        self.image_paths = sorted(glob.glob(root_dir + "test/logical_anomalies/*"))
        self.mask_paths = sorted(
            glob.glob(root_dir + "ground_truth/logical_anomalies/*/000.png")
        )
        self.transform_img = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.transform_large_img = [
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
        ]
        self.transform_large_img = transforms.Compose(self.transform_large_img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        imageo = PIL.Image.open(self.image_paths[idx]).convert("RGB")
        mask = PIL.Image.open(self.mask_paths[idx]).convert("L")
        mask = totensor(mask)
        mask = self.transform_mask(mask)
        totensor = transforms.ToTensor()
        image1 = totensor(imageo)
        image2 = self.transform_large_img(imageo)
        image = self.transform_img(imageo)

        Image = {"image": image, "image1": image1, "image_large": image2}
        return Image


class MVTecLocoTestGoodDataset(Dataset):
    def __init__(self, root_dir, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape

        self.image_paths = sorted(glob.glob(root_dir + "test/good/*"))

        self.transform_img = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.transform_large_img = [
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
        ]
        self.transform_large_img = transforms.Compose(self.transform_large_img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        imageo = PIL.Image.open(self.image_paths[idx]).convert("RGB")
        totensor = transforms.ToTensor()
        image1 = totensor(imageo)
        image2 = self.transform_large_img(imageo)
        image = self.transform_img(imageo)
        mask = torch.zeros((resize_size, resize_size))
        Image = {"image": image, "image1": image1, "image_large": image2, "mask": mask}
        return Image


class MVTecLocoTestStruDataset(Dataset):
    def __init__(self, root_dir, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape

        self.image_paths = sorted(glob.glob(root_dir + "test/structural_anomalies/*"))
        self.mask_paths = sorted(
            glob.glob(root_dir + "ground_truth/structural_anomalies/*/000.png")
        )

        self.transform_img = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.transform_large_img = [
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
        ]
        self.transform_large_img = transforms.Compose(self.transform_large_img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        imageo = PIL.Image.open(self.image_paths[idx]).convert("RGB")
        mask = PIL.Image.open(self.mask_paths[idx]).convert("L")
        mask = totensor(mask)
        mask = self.transform_mask(mask)
        totensor = transforms.ToTensor()
        image1 = totensor(imageo)
        image2 = self.transform_large_img(imageo)
        image = self.transform_img(imageo)
        Image = {"image": image, "image1": image1, "image_large": image2, "mask": mask}
        return Image


class MVTecLocoTestValDataset(Dataset):
    def __init__(self, root_dir, category, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.category = category
        img_path = root_dir + f"{self.category}/*"
        self.image_paths = sorted(glob.glob(img_path + "*.png"))

        self.transform_img = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((self.resize_shape, self.resize_shape)),
            # transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.transform_large_img = [
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
        ]
        self.transform_large_img = transforms.Compose(self.transform_large_img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        imageo = PIL.Image.open(self.image_paths[idx]).convert("RGB")
        totensor = transforms.ToTensor()
        image1 = totensor(imageo)
        image2 = self.transform_large_img(imageo)
        image = self.transform_img(imageo)
        Image = {"image": image, "image1": image1, "image_large": image2}
        return Image
