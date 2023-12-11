#!/bin/bash

mkdir -p data/
cd data/

wget -N https://archive.ics.uci.edu/static/public/371/nips+conference+papers+1987+2015.zip
unzip -n nips+conference+papers+1987+2015.zip -d ./

# https://suitesparse-collection-website.herokuapp.com/SNAP/email-Enron
wget -N http://snap.stanford.edu/data/email-Enron.txt.gz
gzip -d email-Enron.txt.gz

# https://suitesparse-collection-website.herokuapp.com/ND/nd12k
wget -N https://suitesparse-collection-website.herokuapp.com/MM/ND/nd12k.tar.gz
tar -xf nd12k.tar.gz

# https://suitesparse-collection-website.herokuapp.com/Belcastro/human_gene2
wget -N https://suitesparse-collection-website.herokuapp.com/MM/Belcastro/human_gene2.tar.gz
tar -xf human_gene2.tar.gz

# https://suitesparse-collection-website.herokuapp.com/Boeing/ct20stif
wget https://suitesparse-collection-website.herokuapp.com/MM/Boeing/ct20stif.tar.gz
tar -xf ct20stif.tar.gz

# https://suitesparse-collection-website.herokuapp.com/Boeing/pwtk
wget https://suitesparse-collection-website.herokuapp.com/MM/Boeing/pwtk.tar.gz
tar -xf pwtk.tar.gz

# http://sparse.tamu.edu/GHS_psdef/inline_1
wget https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/inline_1.tar.gz
tar -xf inline_1.tar.gz

# http://sparse.tamu.edu/VLSI/vas_stokes_1M
wget https://suitesparse-collection-website.herokuapp.com/MM/VLSI/vas_stokes_1M.tar.gz
tar -xf vas_stokes_1M.tar.gz

# http://sparse.tamu.edu/VLSI/nv2
wget https://suitesparse-collection-website.herokuapp.com/MM/VLSI/nv2.tar.gz
tar -xf nv2.tar.gz

# http://sparse.tamu.edu/SNAP/sx-stackoverflow
wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/sx-stackoverflow.tar.gz

# https://sparse.tamu.edu/Freescale/FullChip
wget https://suitesparse-collection-website.herokuapp.com/MM/Freescale/FullChip.tar.gz

# https://sparse.tamu.edu/HB/bcsstk30
wget https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk30.tar.gz

# https://sparse.tamu.edu/FEMLAB/sme3Db
wget https://suitesparse-collection-website.herokuapp.com/MM/FEMLAB/sme3Db.tar.gz

# https://sparse.tamu.edu/IPSO/TSC_OPF_1047
wget https://suitesparse-collection-website.herokuapp.com/MM/IPSO/TSC_OPF_1047.tar.gz

# https://sparse.tamu.edu/JGD_Groebner/c8_mat11
wget https://suitesparse-collection-website.herokuapp.com/MM/JGD_Groebner/c8_mat11.tar.gz

# https://sparse.tamu.edu/TKK/smt
wget https://suitesparse-collection-website.herokuapp.com/MM/TKK/smt.tar.gz

# https://sparse.tamu.edu/Belcastro/mouse_gene
wget https://suitesparse-collection-website.herokuapp.com/MM/Belcastro/mouse_gene.tar.gz

#cd
#wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
#unzip libtorch-shared-with-deps-latest.zip
#rm libtorch-shared-with-deps-latest.zip
