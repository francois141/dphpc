# SDDMM Project

## Installation

- Install libtorch GPU,
- Build DGL from source and install it (`cmake --install`),
- Update the installation locations in `run_cmake.sh`,
- Make sure `LD_LIBRARY_PATH` contains `/usr/local/lib`,
- Run `./run_cmake`,
- Build and run.

### List of matrices we have until now

#### Small matrices

|   Matrix    |    Rows     |   Cols   |  Non-Zero  | Density |                 Image                  |
|:-----------:|:-----------:|:--------:|:----------:|:-------:|:--------------------------------------:|
|    Fluid    |     656     |     656     |  	18,964 |  4.4%   |    <img src="./images/fluid.png" />    |
|     Oil     |     	66     |  	66  |  	4,356 |  100%   |     <img src="./images/oil.png" />     |
| Biochemical |      	 	1,922      |   	1,922 |  	4,335 |  0.1%   | <img src="./images/biochemical.png" /> |
|   Circuit   |  	 	1,220  |  	1,220 |  	5,860 |  0.39%  |   <img src="./images/circuit.png" />   |
|    Heat     |  	  	1,794  |  	1,794  |  	7,764 |  0.24%  |    <img src="./images/heat.png" />     |
|    Mass     |  	  	420  | 420  |  	 	7,860 |  4.45%  |    <img src="./images/mass.png" />     |
|    Adder    | 	  	1,813  |  1,813  |  	 	11,246 |  0.34%  |    <img src="./images/adder.png" />    |
|  Trackball  |  	   	25,503  |   	25,503  |  	 	15,525 |  0.01%  |  <img src="./images/trackball.png" />  |


#### Dense matrices

|    Matrix     |    Rows    |   Cols   |  Non-Zero  | Density |                  Image                   |
|:-------------:|:----------:|:--------:|:----------:|:-------:|:----------------------------------------:|
| Human Gene 2  |   14,340   |  14,340  | 18,068,388 |  8.8%   |  <img src="./images/human_gene2.png" />  |
|     ND12k     |   36,000   |  36,000  | 14,220,946 |   1%    |     <img src="./images/nd12k.png" />     |
|      Mix      |  	29,957   |     29,957     | 1,990,919 |  0.22%  |      <img src="./images/mix.png" />      |
|   Mecanics    |  	 29,067  |  29,067  | 2,081,063 |  0.24%  |   <img src="./images/mecanics.png" />    |
|     Power     |  	  	8,140  |  	8,140  |  	2,012,833 |  3.03%  |     <img src="./images/power.png" />     |
| Combinatorics |  	  	4,562  |  5,761   |  	2,462,970 |  9.37%  | <img src="./images/combinatorics.png" /> |
|    Stress     |  	  	25,710 |  25,710  |  	3,749,582 |  0.56%  |    <img src="./images/stress.png" />     |
|     Mouse     |  	  45,101 |  45,101  |  	 	28,967,291 |  1.42%  |     <img src="./images/mouse.png" />     |


#### Sparse Matrices

|     Matrix      |   Rows    |   Cols    |  Non-Zero  | Density  |                   Image                    |
|:---------------:|:---------:|:---------:|:----------:|:--------:|:------------------------------------------:|
|   Email enron   |  36,692   |  36,692   |  367,662   |  0.027%  |   <img src="./images/email-enron.png" />   |
|     Boeing      |  52,329   |  52,329   | 2,600,295  |  0.09%   |     <img src="./images/boeing.png" />      |
| Boeing Diagonal |  217,918  |  217,918  | 11,524,432 |  0.02%   | <img src="./images/boeing_diagonal.png" /> |
|    Stiffness    |  503,712  |  503,712  | 36,816,170 |  0.014%  |    <img src="./images/stiffness.png" />    |
| Semi conductor  | 1,090,664 | 1,090,664 | 34,767,207 | 0.0029%  |     <img src="./images/stokes.png" />      |
|      VLSI       | 1,453,908 | 1,453,908 | 37,475,646 | 0.0017%  |      <img src="./images/vlsi.png" />       |
| Stack overflow  | 2,601,977 | 2,601,977 | 36,233,450 | 0.00053% |  <img src="./images/stackoverflow.png" />  |
|      Chip       | 2,987,012 | 2,987,012 | 26,621,983 | 0.00029% |      <img src="./images/chip.png" />       |
