#pragma once
#include <DirectXMath.h>

enum class VertexType : unsigned int {
	ColorShaderVertex = 0
};

struct ColorShaderVertex {
	DirectX::XMFLOAT4 position;
	DirectX::XMFLOAT4 color;
};