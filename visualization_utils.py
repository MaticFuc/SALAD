import matplotlib
matplotlib.use("svg")
import cv2
import numpy as np
from PIL import Image

def save_seg(file_name, file_path, seg, argmax=True, num_cls=6):
    C = num_cls
    seg = seg.squeeze()
    if argmax:
        seg = seg.argmax(dim=0).squeeze().detach().cpu().numpy()
    else:
        seg = seg.squeeze().detach().cpu().numpy()
    seg = seg.astype(np.uint8)

    
    palette = [0, 0, 0, 204, 241, 227, 112, 142, 18, 254, 8, 23, 207, 149, 84, 202, 24, 214,
                    230, 192, 37, 241, 80, 68, 74, 127, 0, 2, 81, 216, 24, 240, 129, 20, 215, 125, 161, 31, 204,
                    254, 52, 116, 117, 198, 203, 4, 41, 68, 127, 252, 61, 21, 3, 142, 40, 10, 159, 241, 61, 36,
                    14, 175, 77, 144, 61, 115, 131, 79, 97, 109, 177, 163, 58, 198, 140, 17, 235, 168, 47, 128, 91,
                    238, 103, 45, 124, 35, 228, 101, 48, 232, 74, 124, 114, 78, 49, 30, 35, 167, 27, 137, 231, 47,
                    235, 32, 39, 56, 112, 32, 62, 173, 79, 86, 44, 201, 77, 47, 217, 246, 223, 57, ]
    palette = palette + [0]*(768-len(palette))
    pi = Image.fromarray(seg, 'P')
    # pi = pi.resize((256,256))
    pi.putpalette(palette)
    pi.save(f"{file_path}/{file_name}")

def save_mask(file_name, file_path, mask):
    mask = mask.squeeze().unsqueeze(-1).detach().cpu().numpy()
    out = mask
    # print(out.shape)
    out = out * 255
    out = out.astype(np.uint8)
    cv2.imwrite(f"{file_path}/{file_name}", out)