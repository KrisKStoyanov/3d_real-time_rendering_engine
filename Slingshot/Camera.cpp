#include "Camera.h"

void Camera::Setup(Transform transform, const float nearPlane, const float farPlane, const float horFov, const float vertFov)
{

	//View
	DirectX::XMFLOAT4 focusPos = DirectX::XMFLOAT4(
		transform.m_Position.x + transform.m_Rotation.x,
		transform.m_Position.y + transform.m_Rotation.y,
		transform.m_Position.z + transform.m_Rotation.z,
		transform.m_Position.w + transform.m_Rotation.w);
	DirectX::XMVECTOR pos = DirectX::XMLoadFloat4(&transform.m_Position);
	DirectX::XMVECTOR focus = DirectX::XMLoadFloat4(&focusPos);
	DirectX::XMVECTOR upDir = DirectX::XMLoadFloat4(&transform.m_UpDir);

	DirectX::XMMatrixLookAtLH(pos, focus, upDir);

	//Projection
	float width = (float)1.0f / tan(horFov * 0.5f);
	float height = (float)1.0f / tan(vertFov * 0.5f);
	float Q = farPlane / (farPlane - nearPlane);

	DirectX::XMFLOAT4X4 tempProj;
	tempProj(0, 0) = width;
	tempProj(1, 1) = height;
	tempProj(2, 2) = Q;
	tempProj(3, 2) = -Q * nearPlane;
	tempProj(2, 3) = 1;
	m_ProjectionMatrix = DirectX::XMLoadFloat4x4(&tempProj);
}
