#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <memory>

#include "utils/util.hpp"
#include "matrices/matrices.h"
#include "utils/random_generator.hpp"

namespace SDDMM {

    template<typename T>
    class Dataset {
        private:
            const std::string name;
        protected:
            Dense<T> A;
            Dense<T> B;

            CSR<T> S_csr;
            COO<T> S_coo;
            
            CSR<T> P_csr;
            COO<T> P_coo;

            Dataset(const std::string &name)
            : name(name)
            {}

            void generateDense(const int M, const int N, const int K) {
                std::vector<std::vector<T>> matrixA(M, std::vector<T>(K));
                for (int m = 0; m < M; m++) { generate_data<T>(matrixA[m], 0, 100); }
                this->A = Dense<T>(matrixA);

                std::vector<std::vector<T>> matrixB(N, std::vector<T>(K));
                for (int n = 0; n < N; n++) { generate_data<T>(matrixB[n], 0, 100); }
                this->B = Dense<T>(matrixB);
            }

        public:

            Dataset(const std::string &name, Dense<T> &A, Dense<T> &B, CSR<T> &S_csr)
            : name(name),
            A(A), B(B),
            S_csr(S_csr), S_coo(S_csr),
            P_csr(), P_coo()
            {}

            Dataset(const std::string &name, Dense<T> &A, Dense<T> &B, COO<T> &S_coo)
            : name(name),
            A(A), B(B),
            S_csr(S_coo), S_coo(S_coo),
            P_csr(), P_coo()
            {}
            
            Dataset(const std::string &name, Dense<T> &A, Dense<T> &B, CSR<T> &S_csr, CSR<T> &P_csr)
            : name(name),
            A(A), B(B),
            S_csr(S_csr), S_coo(S_csr),
            P_csr(P_csr), P_coo(P_csr)
            {}

            Dataset(const std::string &name, Dense<T> &A, Dense<T> &B, COO<T> &S_coo, COO<T> &P_coo)
            : name(name),
            A(A), B(B),
            S_csr(S_coo), S_coo(S_coo),
            P_csr(P_coo), P_coo(P_coo)
            {}


            ~Dataset()
            {}

            const std::string getName() {
                return this->name;
            }

            Dense<T> &getA() {
                return this->A;
            }

            Dense<T> &getB() {
                return this->B;
            }

            CSR<T> &getS_CSR() {
                return this->S_csr;
            }

            COO<T> &getS_COO() {
                return this->S_coo;
            }

            bool hasExpected() {
                return this->P_csr.getRows() > 0 && this->P_coo.getRows() > 0;
            }

            CSR<T> &getExpected_CSR() {
                return this->P_csr;
            }

            void setExpected_CSR(CSR<T> &P_csr) {
                this->P_csr = P_csr;
                this->P_coo = COO<T>(P_csr);
            }

            COO<T> &getExpected_COO() {
                return this->P_coo;
            }
            
            void setExpected_COO(COO<T> &P_coo) {
                this->P_coo = P_coo;
                this->P_csr = CSR<T>(P_coo);
            }
    };

    template<typename T>
    class MatrixMarketDataset : public Dataset<T> {
        private:
            const std::string data_folder;
            const int K;
            const std::string file_name;
            int M;
            int N;

        public:
            MatrixMarketDataset(const std::string &name, const std::string &data_folder, const int K, const std::string &file_name)
            : Dataset<T>(name), data_folder(data_folder), K(K), file_name(file_name)
            {    
                std::vector<Triplet<T>> triplets;

                std::string dataset_path(data_folder);
                dataset_path.append(file_name);

                read_matrix_market(dataset_path, triplets);            

                this->S_csr = CSR<T>(M, N, triplets);
                this->S_coo = COO<T>(M, N, triplets);

                DEBUG_OUT("=== [" << this->getName() << "] Loaded " << triplets.size() << " sparse values from file ===\n" << std::endl);
            }

        private:
            
            // https://math.nist.gov/MatrixMarket/formats.html
            void read_matrix_market(const std::string &dataset_path, std::vector<Triplet<T>> &triplets) {
                FILE* c_file = fopen(dataset_path.c_str(), "r");
                assert(c_file != NULL); // failed to open file

                char buf[256];
                int row, col, triplets_to_load;
                float value;
                
                while (fgets(buf, 256, c_file) != NULL) {
                    if (buf[0] == '%') { continue; }
                    break; // reached non-comment line
                }

                sscanf(buf, "%u %d %d", &M, &N, &triplets_to_load);
                this->generateDense(M, N, K);

                while (fscanf(c_file, "%u %u %f\n", &row, &col, &value) == 3) {
                    row--; col--;
                    triplets.push_back({row, col, value});
                }

                fclose(c_file);
                
                assert(triplets.size() == ((size_t) triplets_to_load));
            }
    };

    class DummyDataset : public Dataset<float> {
        public:

            DummyDataset()
            : Dataset("Dummy")
            {
                std::vector<std::vector<float>> matrixA(2, std::vector<float>(2,1));
                this->A = Dense<float>(matrixA);

                std::vector<std::vector<float>> matrixB(2, std::vector<float>(2,1));
                this->B = Dense<float>(matrixB);

                std::vector<Triplet<float>> S_triplets{{0,0,1}, {0,1,1}, {1,0,1}, {1,1,1}};
                this->S_coo = COO<float>(2, 2, S_triplets);
                this->S_csr = CSR<float>(this->S_coo);

                std::vector<Triplet<float>> P_triplets{{0,0,2}, {0,1,2}, {1,0,2}, {1,1,2}};
                this->P_coo = COO<float>(2, 2, P_triplets);
                this->P_csr = CSR<float>(this->P_coo);
            }

    };
    
    template<typename T>
    class NIPSDataset : public Dataset<T> {
        private:
            const std::string file_name = "NIPS_1987-2015.csv";

        public:

            NIPSDataset(const std::string& data_folder, const int K)
            : Dataset<T>("NIPS"), K(K)
            {
                this->generateDense(M, N, K);

                std::vector<Triplet<T>> triplets;

                std::string dataset_path(data_folder);
                dataset_path.append(file_name);

                std::fstream data_file;
                data_file.open(dataset_path, std::ios::in);

                assert(data_file.is_open()); // failed to open file

                data_file.ignore(61000, '\n'); // skip the long header line

                int i = 0, row = 0, col = 0;

                std::string line;
                while (std::getline(data_file, line)) {
                    std::stringstream lineStream(line);
                    std::string cell;

                    lineStream.ignore(64, ','); // skip the first column (word name)

                    while (std::getline(lineStream, cell, ',')) {
                        if (cell == "0") { col++; continue; }

                        triplets.push_back({ row, col, (T) std::stod(cell) });
                        i++;

                        col++;
                    }
                    row++; col = 0;
                }
                data_file.close();

                this->S_csr = CSR<T>(M, N, triplets);
                this->S_coo = COO<T>(M, N, triplets);

                DEBUG_OUT("=== [" << this->getName() << "] Loaded " << triplets.size() << " sparse values from file ===\n" << std::endl);
            }

        private:
            const int M = 11463;
            const int N = 5811;
            const int K;          
    };

    template<typename T>
    class EMailEnronDataset : public Dataset<T> {
        private:
            const std::string file_name = "email-Enron.txt";

        public:

            EMailEnronDataset(const std::string& data_folder, const int K)
            : Dataset<T>("EMail-Enron"), K(K)
            {   
                
                this->generateDense(M, N, K);

                std::vector<Triplet<T>> triplets;

                std::string dataset_path(data_folder);
                dataset_path.append(file_name);

                std::fstream data_file;
                data_file.open(dataset_path, std::ios::in);

                assert(data_file.is_open()); // failed to open file
            
                for (int i = 0; i < 4; i++) { data_file.ignore(100, '\n'); } // skip the first 4 lines

                int i = 0; // 9 is ASCII value for TAB
                Triplet<T> triplet = { 0, 0, 1 };
                while (data_file >> triplet.row && data_file.ignore(1, 9) && data_file >> triplet.col && data_file.ignore(1, '\n')) {
                    triplets.push_back(triplet);
                    i++;
                }
                data_file.close();

                this->S_csr = CSR<T>(M, N, triplets);
                this->S_coo = COO<T>(M, N, triplets);

                DEBUG_OUT("=== [" << this->getName() << "] Loaded " << triplets.size() << " sparse values from file ===\n" << std::endl);
            }

        private:
            const int M = 36692;
            const int N = 36692;
            const int K;          
    };
    
    template<typename T>
    class ND12KDataset : public MatrixMarketDataset<T> {
        public:
            ND12KDataset(const std::string& data_folder, const int K)
            : MatrixMarketDataset<T>("ND12K", data_folder, K, "nd12k/nd12k.mtx")
            {}
    };


    template<typename T>
    class HumanGene2Dataset : public MatrixMarketDataset<T> {
        public:
            HumanGene2Dataset(const std::string& data_folder, const int K)
            : MatrixMarketDataset<T>("HumanGene2", data_folder, K, "human_gene2/human_gene2.mtx")
            {}
    };

    template<typename T>
    class RandomWithDensityDataset : public Dataset<T> {
        public:

            RandomWithDensityDataset(const int M, const int N, const int K, const double density) : Dataset<T>("RandomWithDensity"), M(M), N(N), K(K), density(density)
            {
                std::clamp(density, 0.0, 1.0);
                assert(0 <= density && density <= 1.0);

                this->generateDense(M, N, K);

                const int nbSamples = density * (M * N);
                std::vector<Triplet<T>> triplets = sampleTriplets<T>(M, N, nbSamples);

                this->S_csr = CSR<T>(M, N, triplets);
                this->S_coo = COO<T>(M, N, triplets);

                DEBUG_OUT("=== [" << this->getName() << "] Loaded " << triplets.size() << " sparse values from random generator ===\n" << std::endl);
            }

        private:
            const int M;
            const int N;
            const int K;
            const double density;
    };

    template<typename T>
    class UnbalancedDataset : public Dataset<T> {
    public:

        UnbalancedDataset(const int M, const int N, const int K, const double density) : Dataset<T>("RandomWithDensity"), M(M), N(N), K(K), density(density)
        {
            std::clamp(density, 0.0, 1.0);
            assert(0 <= density && density <= 1.0);

            this->generateDense(M, N, K);

            const int nbRowsOne = density * M;
            std::vector<Triplet<T>> triplets;
            triplets.reserve(nbRowsOne * N);

            for(int row_idx = 0; row_idx < nbRowsOne;row_idx++) {
                for(int col_idx = 0; col_idx < N;col_idx++) {
                    triplets.push_back(Triplet<T>(row_idx, col_idx, 1.0));
                }
            }

            this->S_csr = CSR<T>(M, N, triplets);
            this->S_coo = COO<T>(M, N, triplets);

            DEBUG_OUT("=== [" << this->getName() << "] Loaded " << triplets.size() << " sparse values from unbalanced generator ===\n" << std::endl);
        }

    private:
        const int M;
        const int N;
        const int K;
        const double density;
    };

    template<typename T>
    class Cage14Dataset : public MatrixMarketDataset<T> {
    public:
        Cage14Dataset(const std::string &data_folder, const int K)
                : MatrixMarketDataset<T>("Cage14", data_folder, K, "cage14/cage14.mtx")
        {}
    };
}
