#include "Light.h"

Light* Light::Create(LIGHT_DESC light_desc)
{
	return new Light(light_desc);
}

LightType Light::GetType()
{
	return m_type;
}

float Light::GetIntensity()
{
	return m_intensity;
}

DirectX::XMFLOAT4 Light::GetColor()
{
	return m_color;
}

Light::Light(LIGHT_DESC light_desc) : 
	m_intensity(light_desc.intensity), 
	m_color(light_desc.color), 
	m_type(light_desc.type)
{

}

