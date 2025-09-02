from glob import glob

import torch
from PIL import Image
from torchvision import transforms
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PseudoLabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasetParameters: dict,
    ):
        super().__init__()
        image_size = datasetParameters["size"]
        path = datasetParameters["path"]

        self.image_size = image_size
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.default = False
        print(path)
        self._files = glob(f"{path}/*_gt.png")
        self._masks = glob(f"{path}/*_refined_seg.png")
        self._files.sort()
        self._masks.sort()

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index: int):
        image_path = self._files[index]
        mask_path = self._masks[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        mask = torch.LongTensor(mask).squeeze()
        return {"image": image, "mask": mask, "index": index}
