#pragma once
#include "Transform.h"

enum class ShadingModel : unsigned int
{
	GoochShading = 0,
	OrenNayarShading
};

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

struct WVPData //vertex exclusive
{
	DirectX::XMMATRIX worldMatrix;
	DirectX::XMMATRIX viewMatrix;
	DirectX::XMMATRIX projMatrix;
};

struct WorldTransformData //pixel exclusive
{
	DirectX::XMVECTOR camPos;
};

struct LightData //pixel exclusive
{
	DirectX::XMVECTOR lightPos;
	DirectX::XMFLOAT4 lightColor;
};

struct MaterialData //pixel exclusive
{
	DirectX::XMFLOAT4 surfaceColor;
	float roughness;
};