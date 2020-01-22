#!/bin/bash

wget -nc -O dataset/OpenImages_images.zip "https://onedrive.live.com/download?cid=B7111B95B80CCC66&resid=B7111B95B80CCC66%2130813&authkey=AHgXVPxKxO_5Fvc"
wget -nc -O dataset/OpenImages_annotations.zip "https://onedrive.live.com/download?cid=B7111B95B80CCC66&resid=B7111B95B80CCC66%2130811&authkey=AMWbBWZVQFbm4jw"
mkdir -p dataset/OpenImages
unzip -d dataset/OpenImages dataset/OpenImages_annotations.zip
unzip -d dataset/OpenImages dataset/OpenImages_images.zip

