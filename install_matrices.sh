#!/bin/bash

mkdir -p data/
cd data/

# Small matrices

# http://sparse.tamu.edu/FIDAP/ex21
wget -N https://suitesparse-collection-website.herokuapp.com/MM/FIDAP/ex21.tar.gz
tar -xf ex21.tar.gz

# http://sparse.tamu.edu/HB/bcsstk02
wget -N https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk02.tar.gz
tar -xf bcsstk02.tar.gz

# http://sparse.tamu.edu/Schulthess/N_biocarta
wget -N https://suitesparse-collection-website.herokuapp.com/MM/Schulthess/N_biocarta.tar.gz
tar -xf N_biocarta.tar.gz

# http://sparse.tamu.edu/Sandia/fpga_dcop_06
wget -N https://suitesparse-collection-website.herokuapp.com/MM/Sandia/fpga_dcop_06.tar.gz
tar -xf fpga_dcop_06.tar.gz

# http://sparse.tamu.edu/Averous/epb0
wget -N https://suitesparse-collection-website.herokuapp.com/MM/Averous/epb0.tar.gz
tar -xf epb0.tar.gz

# http://sparse.tamu.edu/HB/bcsstk07
wget -N https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk07.tar.gz
tar -xf bcsstk07.tar.gz

# http://sparse.tamu.edu/Sandia/adder_dcop_33
wget -N https://suitesparse-collection-website.herokuapp.com/MM/Sandia/adder_dcop_33.tar.gz
tar -xf adder_dcop_33.tar.gz

# http://sparse.tamu.edu/Boeing/bcsstm37
wget -N https://suitesparse-collection-website.herokuapp.com/MM/Boeing/bcsstm37.tar.gz
tar -xf  bcsstm37.tar.gz

# Large matrices

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
tar -xf sx-stackoverflow.tar.gz

# https://sparse.tamu.edu/Freescale/FullChip
wget https://suitesparse-collection-website.herokuapp.com/MM/Freescale/FullChip.tar.gz
tar -xf FullChip.tar.gz

# http://sparse.tamu.edu/POLYFLOW/mixtank_new
wget https://suitesparse-collection-website.herokuapp.com/MM/POLYFLOW/mixtank_new.tar.gz
tar -xf mixtank_new.tar.gz

# https://sparse.tamu.edu/FEMLAB/sme3Db
wget https://suitesparse-collection-website.herokuapp.com/MM/FEMLAB/sme3Db.tar.gz
tar -xf sme3Db.tar.gz

# https://sparse.tamu.edu/IPSO/TSC_OPF_1047
wget https://suitesparse-collection-website.herokuapp.com/MM/IPSO/TSC_OPF_1047.tar.gz
tar -xf TSC_OPF_1047.tar.gz

# https://sparse.tamu.edu/JGD_Groebner/c8_mat11
wget https://suitesparse-collection-website.herokuapp.com/MM/JGD_Groebner/c8_mat11.tar.gz
tar -xf c8_mat11.tar.gz

# https://sparse.tamu.edu/TKK/smt
wget https://suitesparse-collection-website.herokuapp.com/MM/TKK/smt.tar.gz
tar -xf smt.tar.gz

# https://sparse.tamu.edu/Belcastro/mouse_gene
wget https://suitesparse-collection-website.herokuapp.com/MM/Belcastro/mouse_gene.tar.gz
tar -xf mouse_gene.tar.gz

