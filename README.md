# DRP1 tracking on mitochondria in Python

DRP1 tracking on mitochondria done for a Bachelor-thesis at the CECAD from the University of Cologne.
The script takes SIM^2 videos in a TIF format (max 4GB) with the first channel being
the mitochondrial channel and the second channel being the DRP1 channel.

## Usage

The extract.py has to be called `python extract.py -d "<DIR_PATH>"` with `<DIR_PATH>` being the path with
the SIM^2 videos in it. It will then calculate the data and save it to the same folder with the same name
as the TIF file with subfix "_tracking.csv" and "_mitoskel.csv".

## Requirements

This project is using `cupy`, so CUDA has to be installed and an
[CUDA](https://developer.nvidia.com/cuda-toolkit) capable graphics card has to be used.

Videos cannot be bigger than 4GB due to TIF file format restrictions.
