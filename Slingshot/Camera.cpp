#include "Camera.h"

Camera* Camera::Create(CAMERA_DESC* camera_desc)
{
	return new Camera(camera_desc);
}

DirectX::XMMATRIX Camera::GetViewMatrix()
{
	return m_viewMatrix;
}

DirectX::XMMATRIX Camera::GetProjectionMatrix()
{
	return m_projectionMatrix;
}

void Camera::OnUpdate()
{
	m_pTransform->Rotate(m_pTransform->GetRotation());
	m_pTransform->OnUpdate();
}

Transform* Camera::GetTransform()
{
	return m_pTransform;
}


Camera::Camera(CAMERA_DESC* camera_desc) :
	m_viewMatrix(), m_projectionMatrix(), m_pTransform(nullptr)
{
	camera_desc->transform_desc != nullptr ?
		m_pTransform = Transform::Create(camera_desc->transform_desc) :
		m_pTransform = Transform::Create(&TRANSFORM_DESC());

	m_viewMatrix = DirectX::XMMatrixLookAtLH(
		m_pTransform->GetPosition(), 
		DirectX::XMVectorAdd(m_pTransform->GetPosition(), m_pTransform->GetForwardDir()),
		m_pTransform->GetUpDir());

	m_projectionMatrix = DirectX::XMMatrixPerspectiveFovLH(
		DirectX::XMConvertToRadians(camera_desc->verticalFovAngle),
		camera_desc->lenseWidth / camera_desc->lenseHeight,
		camera_desc->nearClipDist,
		camera_desc->farClipDist);
}
