#!/bin/bash

mkdir -p data/
cd data/

wget -nc https://archive.ics.uci.edu/static/public/371/nips+conference+papers+1987+2015.zip
unzip -n nips+conference+papers+1987+2015.zip -d ./

wget -nc http://snap.stanford.edu/data/email-Enron.txt.gz
gzip -d 1138_bus.tar.gz

# Download test matrix market format
wget https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz
gzip -d 1138_bus.tar.gz

# Download big matrix market format
# From : https://sparse.tamu.edu/vanHeukelum
wget https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage14.tar.gz
gzip -d cage14.tar.gz

cd ..

#cd
#wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
#unzip libtorch-shared-with-deps-latest.zip
#rm libtorch-shared-with-deps-latest.zip
