#pragma once
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <Windows.h>
#include <shellapi.h>

#include <windef.h>
#include <windowsx.h>

#include <wrl.h>
#include <wrl/client.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <vector>

enum class GraphicsContextType : unsigned int {
	D3D11 = 0,
	D3D12,
	OpenGL,
	Vulkan
};

class GraphicsContext
{
public:
	//Render
	//Update
	//Shutdown
	//Load?
};

