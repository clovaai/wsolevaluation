#!/bin/bash

wget -nc -P dataset/ http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
wget -nc -O dataset/CUBV2.tar "https://onedrive.live.com/download?cid=B7111B95B80CCC66&resid=B7111B95B80CCC66%2130812&authkey=AFMzb4akufUiWU0"
mkdir -p dataset/CUB_200_2011
tar xvf dataset/CUB_200_2011.tgz -C dataset/
mv dataset/CUB_200_2011/images dataset/CUB && rm -rf dataset/CUB_200_2011
tar xvf dataset/CUBV2.tar -C dataset/CUB