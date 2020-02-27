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
	float lightInt;
};

enum class ShadingModel : unsigned int {
	GoochShading = 0,
};

struct Vertex {

};

struct GoochShadingVertex : public Vertex {
	DirectX::XMFLOAT4 position;
	DirectX::XMFLOAT4 color;
	DirectX::XMFLOAT4 normal;
};