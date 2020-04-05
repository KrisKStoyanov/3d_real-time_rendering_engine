#pragma once
#include "DirectXMath.h"

enum LightType {
	Directional = 0,
	Point,
	Spotlight
};

struct LIGHT_DESC 
{
	LightType type = LightType::Point;
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
	Light(LIGHT_DESC light_desc);
	LightType m_type;
	float m_intensity; 
	DirectX::XMFLOAT4 m_color;

	//Photon mapping
	float m_power;
	float m_photonEmission; //num of photons emitted
	float m_powerPerPhoton; //power/photonEmission
};
