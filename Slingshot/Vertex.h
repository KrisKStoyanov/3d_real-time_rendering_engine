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
	DirectX::XMFLOAT2 uv;
};

enum class ShaderType : unsigned int
{
	VERTEX_SHADER = 0,
	PIXEL_SHADER
};

struct PLANE_DESC
{
	float width, length;
};

struct CUBE_DESC
{
	float width, height, length;
};

struct SPHERE_DESC
{
	int slices, stacks;
	float radius;
};

struct TRIANGLE_DESC
{
	float p0_x, p0_y, p0_z;
	float p1_x, p1_y, p1_z;
	float p2_x, p2_y, p2_z;

	float n_x, n_y, n_z;
};