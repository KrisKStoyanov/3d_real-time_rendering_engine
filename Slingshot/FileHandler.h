#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <io.h>
#include <iostream>

#include <d3dcompiler.h>

std::vector<uint8_t> ReadFile(std::string filePath);
uintmax_t ComputeFileSize(std::string filePath);