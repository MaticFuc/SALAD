import torch
import torch.nn as nn

import composition_map_creation.dino.vision_transformer as vits


class DinoFeaturizer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.dim = 70
        patch_size = 8 
        self.n_feats = 768
        self.patch_size = patch_size
        self.feat_type = "feat"
        arch = "vit_base"
        self.model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=0.1)
        self.whetherdropout = False

        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/" + url
        )
        self.model.load_state_dict(state_dict, strict=True)

        

    def forward(self, img, n=1):
        self.model.eval()
        with torch.no_grad():
            assert img.shape[2] % self.patch_size == 0
            assert img.shape[3] % self.patch_size == 0

            feat, _, _ = self.model.get_intermediate_feat(img, n=n)
            feat = feat[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            image_feat = (
                feat[:, 1:, :]
                .reshape(feat.shape[0], feat_h, feat_w, -1)
                .permute(0, 3, 1, 2)
            )
        

        code = image_feat

        return image_feat, code
