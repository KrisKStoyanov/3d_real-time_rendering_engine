#pragma once
#include "Transform.h"

class Camera
{
public:
	Camera() {}

	void Setup(
		Transform transform, 
		const float nearPlane, 
		const float farPlane,
		const float horFov,
		const float vertFov);

	Transform m_Transform;

	DirectX::XMFLOAT3 m_Target;

	DirectX::XMMATRIX m_ViewMatrix = DirectX::XMMatrixIdentity();
	DirectX::XMMATRIX m_ProjectionMatrix = DirectX::XMMatrixIdentity();
};

