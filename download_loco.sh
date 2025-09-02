#!/usr/bin/env bash
mkdir data
cd data
mkdir mvtec_loco
cd mvtec_loco
wget https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz
tar -xvf mvtec_loco_anomaly_detection.tar.xz
rm mvtec_loco_anomaly_detection.tar.xz
cd ..
cd ..