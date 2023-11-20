#!/bin/bash

mkdir -p data/
cd data/

wget -N https://archive.ics.uci.edu/static/public/371/nips+conference+papers+1987+2015.zip
unzip -n nips+conference+papers+1987+2015.zip -d ./

wget -N http://snap.stanford.edu/data/email-Enron.txt.gz
gzip -d email-Enron.txt.gz

# Download test matrix market format
# wget https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz
# tar -xf 1138_bus.tar.gz

# Download big matrix market format from: https://sparse.tamu.edu/vanHeukelum
wget -N https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage14.tar.gz
tar -xf cage14.tar.gz

cd ..

#cd
#wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
#unzip libtorch-shared-with-deps-latest.zip
#rm libtorch-shared-with-deps-latest.zip
