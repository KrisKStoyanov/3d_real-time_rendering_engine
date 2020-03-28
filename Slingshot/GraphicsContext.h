#pragma once
#include "Window.h"
#include "DirectXMath.h"

namespace gfx
{
	enum class ContextType : unsigned int
	{
		D3D11 = 0,
		D3D12,
		OpenGL,
		Vulkan
	};

	enum class TopologyType : unsigned int
	{
		TRIANGLESTRIP = 0,
		TRIANGLELIST,
		LINESTRIP,
		LINELIST,
		POINTLIST,
		PATCHLIST
	};

	struct WVPData
	{
		DirectX::XMMATRIX worldMatrix;
		DirectX::XMMATRIX viewMatrix;
		DirectX::XMMATRIX projMatrix;
	};

	struct WorldTransformData
	{
		DirectX::XMVECTOR camPos;
		DirectX::XMVECTOR lightPos;
	};

	struct LightData
	{
		DirectX::XMFLOAT4 lightColor;
	};

	struct MaterialData
	{
		DirectX::XMFLOAT4 surfaceColor;
		float roughness;
	};
}

//TODO:
//virtual abstract class for extending functionality across different context types through polymorphism