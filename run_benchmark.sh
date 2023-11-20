#!/bin/bash

### When running this script, make sure that the DUMP macro is defined in util.hpp ###

mkdir -p ./plots/results/

touch ./plots/results/results.csv
true > ./plots/results/results.csv

./build/src/dphpc --K 32 --data_folder data/ >> ./plots/results/results.csv
./build/src/dphpc --K 64 --data_folder data/ >> ./plots/results/results.csv --no_csv_header
./build/src/dphpc --K 128 --data_folder data/ >> ./plots/results/results.csv --no_csv_header

for repeats in {2..5}
do
    ./build/src/dphpc --K 32 --data_folder data/ >> ./plots/results/results.csv --no_csv_header
    ./build/src/dphpc --K 64 --data_folder data/ >> ./plots/results/results.csv --no_csv_header
    ./build/src/dphpc --K 128 --data_folder data/ >> ./plots/results/results.csv --no_csv_header
done

