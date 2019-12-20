#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <io.h>
#include <iostream>

#include <d3dcompiler.h>

bool GetBytecode(std::string filePath, std::vector<uint8_t>* bytecode);
uintmax_t ComputeFileSize(std::string filePath);