#include "FileParsing.h"

char* GetBytecode(const char* filename, size_t& filesize)
{
	std::ifstream shaderFileStream;
	shaderFileStream.open(filename, std::ios::in | std::ios::binary | std::ios::ate);
	if (shaderFileStream.is_open()) {
		filesize = static_cast<size_t>(shaderFileStream.tellg());
		shaderFileStream.seekg(0, std::ios::beg);
		char* buffer = new char[filesize];
		shaderFileStream.read(buffer, filesize);
		shaderFileStream.close();
		return buffer;
	}
	return nullptr;
}