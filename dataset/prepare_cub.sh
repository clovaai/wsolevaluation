#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45" -nc -O dataset/CUB_200_2011.tgz && rm -rf /tmp/cookies.txt
wget -nc -O dataset/CUBV2.tar "https://onedrive.live.com/download?cid=B7111B95B80CCC66&resid=B7111B95B80CCC66%2130812&authkey=AFMzb4akufUiWU0"
mkdir -p dataset/CUB_200_2011
tar xvf dataset/CUB_200_2011.tgz -C dataset/
mv dataset/CUB_200_2011/images dataset/CUB && rm -rf dataset/CUB_200_2011
tar xvf dataset/CUBV2.tar -C dataset/CUB
