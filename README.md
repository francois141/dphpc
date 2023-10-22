## Requirements

You need to install the C++ distributions of pytorch as indicated [here](https://pytorch.org/cppdocs/installing.html). The current way of compilation expects you to unzip the downloaded file in your homefolder, by setting `DCMAKE_PREFIX_PATH` accordingly. You can run the commands below:

```
cd
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
rm libtorch-shared-with-deps-latest.zip
```