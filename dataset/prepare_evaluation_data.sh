#!/bin/bash

wget -nc -P dataset/ https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-threshold0.7.tar.gz
mkdir -p dataset/ILSVRC/val
tar xvf dataset/ILSVRC2012_img_val.tar -C dataset/ILSVRC/val
tar xvf dataset/imagenetv2-threshold0.7.tar.gz -C dataset/ILSVRC/
mv dataset/ILSVRC/imagenetv2-threshold0.7 dataset/ILSVRC/val2

wget -nc -P dataset/ http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
wget -nc -O dataset/CUBV2.tar "https://onedrive.live.com/download?cid=B7111B95B80CCC66&resid=B7111B95B80CCC66%2130812&authkey=AFMzb4akufUiWU0"
mkdir -p dataset/CUB_200_2011
tar xvf dataset/CUB_200_2011.tgz -C dataset/
mv dataset/CUB_200_2011/images dataset/CUB && rm -rf dataset/CUB_200_2011
tar xvf dataset/CUBV2.tar -C dataset/CUB

wget -nc -O dataset/OpenImages30k_eval.zip "https://onedrive.live.com/download?cid=B7111B95B80CCC66&resid=B7111B95B80CCC66%2130826&authkey=AJI0Cb0e74L3s00"
mkdir -p dataset/OpenImages
unzip -d dataset/OpenImages dataset/OpenImages30k_eval.zip

