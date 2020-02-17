#include "Camera.h"

Camera* Camera::Create(CAMERA_DESC* camera_desc, Transform* transform)
{
	return new Camera(camera_desc, transform);
}

DirectX::XMMATRIX Camera::GetViewMatrix()
{
	return m_viewMatrix;
}

DirectX::XMMATRIX Camera::GetProjectionMatrix()
{
	return m_projectionMatrix;
}

void Camera::OnFrameRender(Transform* transform)
{
	m_viewMatrix = DirectX::XMMatrixLookAtLH(
		transform->GetPosition(),
		DirectX::XMVectorAdd(transform->GetPosition(), transform->GetForwardDir()),
		transform->GetUpDir());
}

Camera::Camera(CAMERA_DESC* camera_desc, Transform * transform) :
	m_viewMatrix(), m_projectionMatrix()
{
	m_viewMatrix = DirectX::XMMatrixLookAtLH(
		transform->GetPosition(),
		DirectX::XMVectorAdd(transform->GetPosition(), transform->GetForwardDir()),
		transform->GetUpDir());

	m_projectionMatrix = DirectX::XMMatrixPerspectiveFovLH(
		DirectX::XMConvertToRadians(camera_desc->verticalFovAngle),
		camera_desc->lenseWidth / camera_desc->lenseHeight,
		camera_desc->nearClipDist,
		camera_desc->farClipDist);
}
