#pragma once
#include "Transform.h"

struct VS_CONSTANT_BUFFER
{
	DirectX::XMMATRIX worldMatrix;
	DirectX::XMMATRIX viewMatrix;
	DirectX::XMMATRIX projMatrix;
};

struct PS_CONSTANT_BUFFER
{
	DirectX::XMVECTOR camPos;
	DirectX::XMVECTOR lightPos;
	DirectX::XMFLOAT4 lightColor;
	DirectX::XMFLOAT4 surfaceColor;
	float roughness;
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