#include "FileHandler.h"

bool GetBytecode(std::string filePath, std::vector<uint8_t>* bytecode)
{
	std::ifstream ifs;
	ifs.open(filePath, std::ifstream::in | std::ifstream::binary);
	if (ifs.good()) {
		uintmax_t ifsSize = ComputeFileSize(filePath);
		bytecode->resize(ifsSize);
		ifs.seekg(0, std::ios::beg);
		ifs.read(reinterpret_cast<char*>(bytecode[0].data()), ifsSize);
		ifs.close();
		return true;
	}
	return false;
}

uintmax_t ComputeFileSize(std::string filePath)
{
	std::filesystem::path p(filePath);
	if (std::filesystem::exists(p) && std::filesystem::is_regular_file(p)) {
		return std::filesystem::file_size(p);
	}
	return 0;
}