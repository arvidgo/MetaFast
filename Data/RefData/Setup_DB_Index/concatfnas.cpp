#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <filesystem>
#include <sstream>
namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    if(!(argc == 5 || argc == 6)) {
        std::cout << "Usage: ./concatfnas.o fnaListFile fnaPath outDirPath targetsize[0=concat all into 1 file] (GB, ~mmisize/6.3) [silentmode=0]" << std::endl;
        return 1;
    }
    const std::string iPath = argv[1];
    const std::string dPath = argv[2];
    const std::string oPath = argv[3];
    const unsigned long MAX_SIZE_GIGABYTES = 1000000000 * std::stof(argv[4]);
    bool debugmode = (argc == 6 && std::stoi(argv[5]) == 1);
    unsigned int fCount = 1;
    std::string filename;

    std::ifstream fnaListFile(iPath);
    if(!fnaListFile.is_open()) {
        std::cout << "fnaListFile file not found" << std::endl;
        return 1;
    }
    std::ofstream ofile(fs::path(oPath) / ("file" + std::to_string(fCount) + ".fna.gz"), std::ios_base::binary);
    std::ofstream content(fs::path(oPath) / "contentinfo.txt");
    content << "#file " << fCount << "\n";
    std::size_t currentFile = 0;
    unsigned long currentSize = 0;

    while(getline(fnaListFile, filename)) {
        if(MAX_SIZE_GIGABYTES != 0 && currentSize >= MAX_SIZE_GIGABYTES) {
            ofile.close();
            ofile.clear();
            ++fCount;
            std::cout << "NEW FILE NOW - FILE NR: " << fCount << std::endl;
            content << "#file " << fCount << "\n";
            ofile.open(fs::path(oPath) / ("file" + std::to_string(fCount) + ".fna.gz"));
            currentSize = 0;
        }
        fs::path currPath = fs::path(dPath) / filename;
        std::ifstream ifile(currPath, std::ios_base::binary);
        ofile << ifile.rdbuf();
        currentSize += fs::file_size(currPath);
        ++currentFile;
        if(debugmode)
            std::cout << "[" << currentFile << "] " <<  currPath << "\n";
        ifile.close();
        content << filename << "\n";

    }
    std::cout << "done concatenating" << std::endl;
    return 0;
}