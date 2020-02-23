#include "Camera.h"

Camera* Camera::Create(CAMERA_DESC& camera_desc, Transform& transform)
{
	return new Camera(camera_desc, transform);
}

Camera::Camera(CAMERA_DESC& camera_desc, Transform& transform) :
	m_viewMatrix(), m_projectionMatrix(),
	m_lastMouseX(0.0f), m_lastMouseY(0.0f),
	m_rotationSensitivity(camera_desc.rotationSensitivity),
	m_rotate(false)
{
	m_viewMatrix = DirectX::XMMatrixLookAtLH(
		transform.GetPosition(),
		DirectX::XMVectorAdd(transform.GetPosition(), transform.GetForwardDir()),
		transform.GetUpDir());

	m_projectionMatrix = DirectX::XMMatrixPerspectiveFovLH(
		DirectX::XMConvertToRadians(camera_desc.verticalFovAngle),
		camera_desc.lenseWidth / camera_desc.lenseHeight,
		camera_desc.nearClipDist,
		camera_desc.farClipDist);
}

DirectX::XMMATRIX Camera::GetViewMatrix()
{
	return m_viewMatrix;
}

DirectX::XMMATRIX Camera::GetProjectionMatrix()
{
	return m_projectionMatrix;
}

void Camera::GetMouseCoord(float& mouseX, float& mouseY)
{
	mouseX = m_lastMouseX;
	mouseY = m_lastMouseY;
}

void Camera::SetMouseCoord(float mouseX, float mouseY)
{
	m_lastMouseX = mouseX;
	m_lastMouseY = mouseY;
}

float Camera::GetRotationSensitivity()
{
	return m_rotationSensitivity;
}

bool Camera::GetRotateStatus()
{
	return m_rotate;
}

void Camera::SetRotateStatus(bool rotate)
{
	m_rotate = rotate;
}

void Camera::OnFrameRender(Transform& transform)
{
	m_viewMatrix = DirectX::XMMatrixLookAtLH(
		transform.GetPosition(),
		DirectX::XMVectorAdd(transform.GetPosition(), transform.GetForwardDir()),
		transform.GetUpDir());
}
