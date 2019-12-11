//Windows(OS) Headers
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <shellapi.h>

#if defined(min)
#undef min
#endif

#if defined(max)
#undef max
#endif

#if defined(CreateWindow)
#undef CreateWindow
#endif

#include <wrl.h>

//D3D12 Headers
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>

//D3D12 Extension Library
#include <d3dx12.h>

//Standard Template Library
#include <algorithm>
#include <cassert>
#include <chrono>

//-----
#include "Helpers.h"