#pragma once
#include "Vertex.h"
#include "GraphicsContext.h"

struct MATERIAL_DESC
{
	ShadingModel shadingModel;
	DirectX::XMFLOAT4 surfaceColor;
	float roughness;
	float diffuseReflectionCoefficient;
	float specularReflectionCoefficient;
	float transmitCoefficient;
	float absorbCoefficient;
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
	float m_diffuseReflectionCoefficient;
	float m_specularReflectionCoefficient;
	float m_transmitCoefficient;
	float m_absorbCoefficient;
};
