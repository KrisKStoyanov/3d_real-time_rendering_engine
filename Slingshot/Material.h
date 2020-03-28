#pragma once
#include "Vertex.h"

enum class ShadingModel : unsigned int 
{
	GoochShading = 0,
	OrenNayarShading
};

struct MATERIAL_DESC
{
	ShadingModel shadingModel;
	DirectX::XMFLOAT4 surfaceColor;
	float roughness;
};

class Material
{
public:
	static Material* Create(MATERIAL_DESC mat_desc);
	inline ShadingModel GetShadingModel() { return m_shadingModel; }

	inline DirectX::XMFLOAT4 GetSurfaceColor() { return m_surfaceColor; }
	inline float GetRoughness() { return m_roughness; }
private:
	Material(MATERIAL_DESC mat_desc);
	ShadingModel m_shadingModel;
	DirectX::XMFLOAT4 m_surfaceColor;
	float m_roughness;
};
