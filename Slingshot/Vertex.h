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

enum class ShaderType : unsigned int
{
	VERTEX_SHADER = 0,
	PIXEL_SHADER
};

struct CBufferData
{

};

struct WVPData : public CBufferData
{
	DirectX::XMMATRIX worldMatrix;
	DirectX::XMMATRIX viewMatrix;
	DirectX::XMMATRIX projMatrix;
};

struct WorldTransformData : public CBufferData
{
	DirectX::XMVECTOR camPos;
	DirectX::XMVECTOR lightPos;
};

struct LightData : public CBufferData
{
	DirectX::XMFLOAT4 lightColor;
};

struct MaterialData : public CBufferData
{
	DirectX::XMFLOAT4 surfaceColor;
	float roughness;
};