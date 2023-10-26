#!/bin/bash

mkdir -p data/
cd data/

wget -nc https://archive.ics.uci.edu/static/public/371/nips+conference+papers+1987+2015.zip
unzip -n nips+conference+papers+1987+2015.zip -d ./

wget -nc http://snap.stanford.edu/data/email-Enron.txt.gz
gzip -d email-Enron.txt.gz

cd ..

cd
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
rm libtorch-shared-with-deps-latest.zip