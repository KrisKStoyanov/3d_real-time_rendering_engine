#include "Material.h"

Material::Material(MATERIAL_DESC mat_desc)
{
	m_shadingModel = mat_desc.shadingModel;
	m_surfaceColor = mat_desc.surfaceColor;
}

ShadingModel Material::GetShadingModel()
{
	return m_shadingModel;
}

DirectX::XMFLOAT4 Material::GetSurfaceColor()
{
	return m_surfaceColor;
}

