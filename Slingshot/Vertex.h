#pragma once
#include "Transform.h"

enum class ShadingModel : unsigned int
{
	GoochShading = 0,
	OrenNayarShading,
	FinalGathering
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

struct FinalGatheringVertex : public Vertex
{
	DirectX::XMFLOAT4 normal;
};

enum class ShaderType : unsigned int
{
	VERTEX_SHADER = 0,
	PIXEL_SHADER,
	COMPUTE_SHADER
};