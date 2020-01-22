#!/bin/bash

wget -nc -P dataset/ https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-threshold0.7.tar.gz
mkdir -p dataset/ILSVRC/val
tar xvf dataset/ILSVRC2012_img_val.tar -C dataset/ILSVRC/val
tar xvf dataset/imagenetv2-threshold0.7.tar.gz -C dataset/ILSVRC/
mv dataset/ILSVRC/imagenetv2-threshold0.7 dataset/ILSVRC/val2
mkdir -p dataset/ILSVRC/train
tar xvf dataset/ILSVRC2012_img_train.tar -C dataset/ILSVRC/train
cd dataset/ILSVRC/train
find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'