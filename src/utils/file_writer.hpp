#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

namespace SDDMM {

    class Output {
    public:
        const std::string path;

        Output(const std::string &path) : path(path) {
            // Create and open a text file
            file.open(path);
        }

        ~Output() {
            file.close();
        }

        virtual void writeLine(std::string, std::string, std::string) = 0;

    protected:
        std::ofstream file;
    };

    class CSVWriter : public Output{
    public:
        const std::string header;
        CSVWriter(const std::string &path, const std::string &header) : Output(path), header(header) {
            this->file << header << "\n";
        }

        void writeLine(const std::string algorithm, const std::string size, const std::string time) override {
            this->file << algorithm << "," << size << "," << time << std::endl;
        }
    };

}