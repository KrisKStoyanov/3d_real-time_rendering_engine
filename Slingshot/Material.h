#pragma once
#include "Vertex.h"

enum class ShadingModel : unsigned int 
{
	GoochShading = 0,
};

struct MATERIAL_DESC
{
	ShadingModel shadingModel;
	DirectX::XMFLOAT4 surfaceColor;
};

class Material
{
public:
	Material(MATERIAL_DESC mat_desc);
	ShadingModel GetShadingModel();
	DirectX::XMFLOAT4 GetSurfaceColor();
private:
	DirectX::XMFLOAT4 m_surfaceColor;
	ShadingModel m_shadingModel;
};
