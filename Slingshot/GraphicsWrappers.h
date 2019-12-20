#pragma once
#include <DirectXMath.h>

struct Vertex {
	DirectX::XMFLOAT4 position;
	DirectX::XMFLOAT4 color;
};

struct VS_ConstantBuffer
{
	DirectX::XMFLOAT4X4 matWrapper;
	DirectX::XMFLOAT4 vecWrapper;
	float fWrapperA;
	float fWrapperB;
	float fWrapperC;
	float fCountWrapper;
};

struct PS_ConstantBuffer
{
	DirectX::XMFLOAT4X4 matWrapper;
	DirectX::XMFLOAT4 vecWrapper;
	float fWrapperA;
	float fWrapperB;
	float fWrapperC;
	float fCountWrapper;
};