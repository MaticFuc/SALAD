#!/usr/bin/env bash
mkdir pretrained_models
cd pretrained_models
pip install gdown
gdown https://docs.google.com/uc?id=1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..