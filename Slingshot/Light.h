#pragma once
#include "DirectXMath.h"

enum LightType {
	Directional = 0,
	Point,
	Spotlight
};

struct LIGHT_DESC 
{
	LightType type = LightType::Directional;
	float intensity = 1.0f;
	DirectX::XMFLOAT4 color = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
};

class Light {
public:
	static Light* Create(LIGHT_DESC light_desc);
	LightType GetType();
	float GetIntensity();
	DirectX::XMFLOAT4 GetColor();
private:
	Light(LIGHT_DESC light_desc) : 
		m_intensity(light_desc.intensity),
		m_color(light_desc.color),
		m_type(light_desc.type) {}
	LightType m_type;
	float m_intensity;
	DirectX::XMFLOAT4 m_color;
};
