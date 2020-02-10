#pragma once
#include "Window.h"

enum class GraphicsContextType : unsigned int {
	D3D11 = 0,
	D3D12,
	OpenGL,
	Vulkan
};
