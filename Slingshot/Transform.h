#pragma once
#include <DirectXMath.h>

class Transform
{
public:
	Transform();
	~Transform();

	DirectX::XMFLOAT4 m_Position = DirectX::XMFLOAT4(0.0f, 0.0f, 0.0f, 1.0f);
	DirectX::XMFLOAT4 m_Rotation = DirectX::XMFLOAT4(0.0f, 0.0f, 0.0f, 1.0f);
	DirectX::XMFLOAT4 m_Scale = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);

	DirectX::XMFLOAT4 m_UpDir = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);
private:

};

