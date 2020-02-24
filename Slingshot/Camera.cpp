#include "Camera.h"

Camera* Camera::Create(CAMERA_DESC& camera_desc, Transform& transform)
{
	return new Camera(camera_desc, transform);
}

Camera::Camera(CAMERA_DESC& camera_desc, Transform& transform) :
	m_viewMatrix(), m_projectionMatrix(),
	m_lastMouseX(0), m_lastMouseY(0),
	m_rotationSpeed(camera_desc.rotationSpeed),
	m_rotate(false), m_translationSpeed(camera_desc.translationSpeed)
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

void Camera::GetMouseCoord(int& mouseX, int& mouseY)
{
	mouseX = m_lastMouseX;
	mouseY = m_lastMouseY;
}

void Camera::SetMouseCoord(int mouseX, int mouseY)
{
	m_lastMouseX = mouseX;
	m_lastMouseY = mouseY;
}

float Camera::GetRotationSpeed()
{
	return m_rotationSpeed;
}

bool Camera::GetRotateStatus()
{
	return m_rotate;
}

void Camera::SetRotateStatus(bool rotate)
{
	m_rotate = rotate;
}

float Camera::GetTranslationSpeed()
{
	return m_translationSpeed;
}

void Camera::OnFrameRender(Transform& transform)
{
	m_viewMatrix = DirectX::XMMatrixLookAtLH(
		transform.GetPosition(),
		DirectX::XMVectorAdd(transform.GetPosition(), transform.GetForwardDir()),
		transform.GetUpDir());
}
