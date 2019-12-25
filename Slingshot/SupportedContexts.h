#pragma once

//Graphics Context
enum class GC : unsigned int {
	D3D11 = 0,
	D3D12,
	OpenGL,
	Vulkan
};

//Interop Context
enum class IC : unsigned int {
	CUDA = 0,
	OptiX,
	OpenCL
};