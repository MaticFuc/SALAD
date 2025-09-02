#!/usr/bin/env bash
mkdir data
cd data
pip install gdown
gdown https://docs.google.com/uc?id=1yXdOhHLO47wl7pGYjUudvSRsCz84E-Gc
unzip mvtec_loco_composition_maps.zip
rm mvtec_loco_composition_maps.zip
cd ..