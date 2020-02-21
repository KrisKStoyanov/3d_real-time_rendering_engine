#pragma once
#include "Transform.h"

struct VS_CONSTANT_BUFFER
{
	DirectX::XMMATRIX wvpMatrix;
	VS_CONSTANT_BUFFER(
		DirectX::XMMATRIX _wvpMatrix = DirectX::XMMatrixIdentity()) : 
		wvpMatrix(_wvpMatrix) {}
};

enum class VertexType : unsigned int {
	ColorShaderVertex = 0
};

struct Vertex {

};

struct ColorShaderVertex : public Vertex {
	DirectX::XMFLOAT4 position;
	DirectX::XMFLOAT4 color;
};