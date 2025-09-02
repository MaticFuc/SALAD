#!/usr/bin/env bash
mkdir results
cd results
pip install gdown
gdown https://docs.google.com/uc?id=1KhhzHw6-oP5H14e0gG_DkXpQijM9_M0D
unzip salad_loco.zip
rm salad_loco.zip
cd ..