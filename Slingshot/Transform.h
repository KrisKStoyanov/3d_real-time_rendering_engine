#pragma once
#include <DirectXMath.h>

struct TRANSFORM_DESC {

};

class Transform
{
public:
	Transform();
	Transform(TRANSFORM_DESC* transform_desc);
	~Transform();

	void Shutdown();

	DirectX::XMFLOAT4 m_position = DirectX::XMFLOAT4(0.0f, 0.0f, 0.0f, 1.0f);
	DirectX::XMFLOAT4 m_rotation = DirectX::XMFLOAT4(0.0f, 0.0f, 0.0f, 1.0f);
	DirectX::XMFLOAT4 m_scale = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);

	DirectX::XMFLOAT4 m_upDir = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);
private:
	TRANSFORM_DESC* m_pDesc;
};

