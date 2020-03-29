#pragma once
#include "Transform.h"

enum class Topology : unsigned int
{
	TRIANGLESTRIP = 0,
	TRIANGLELIST,
	LINESTRIP,
	LINELIST,
	POINTLIST,
	PATCHLIST //pending tessalation implementation
};

struct Vertex 
{
	DirectX::XMFLOAT4 position;
};

struct GoochShadingVertex : public Vertex 
{
	DirectX::XMFLOAT4 normal;
};

struct OrenNayarVertex : public Vertex
{
	DirectX::XMFLOAT4 normal;
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