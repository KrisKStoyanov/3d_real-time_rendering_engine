#include "Transform.h"

Transform* Transform::Create(TRANSFORM_DESC& transform_desc)
{
	return new Transform(transform_desc);
}

Transform::Transform(TRANSFORM_DESC& transform_desc) :
	m_worldMatrix(DirectX::XMMatrixIdentity()),
	m_translatioMatrix(DirectX::XMMatrixIdentity()), 
	m_rotationMatrix(DirectX::XMMatrixIdentity()), 
	m_scalingMatrix(DirectX::XMMatrixIdentity()),
	m_position(), m_rotation(), m_scale(),
	m_forwardDir(), m_rightDir(), m_upDir(),
	m_positionDynamic(),
	m_rotationDynamic(),
	m_scaleDynamic()
{

	m_position = DirectX::XMVectorSet(
		transform_desc.position.x,
		transform_desc.position.y,
		transform_desc.position.z,
		transform_desc.position.w);

	m_positionDynamic = DirectX::XMFLOAT4(
		transform_desc.position.x,
		transform_desc.position.y,
		transform_desc.position.z,
		transform_desc.position.w);

	//Used in Radians
	m_rotation = DirectX::XMVectorSet(
		0.0f,
		0.0f,
		0.0f,
		0.0f);

	//Used in Degrees
	m_rotationDynamic = DirectX::XMFLOAT4(
		0.0f,
		0.0f,
		0.0f,
		0.0f);

	m_scale = DirectX::XMVectorSet(
		transform_desc.scale.x,
		transform_desc.scale.y,
		transform_desc.scale.z,
		transform_desc.scale.w);

	m_scaleDynamic = DirectX::XMFLOAT4(
		transform_desc.scale.x,
		transform_desc.scale.y,
		transform_desc.scale.z,
		transform_desc.scale.w);

	m_forwardDir = DirectX::XMVectorSet(
		transform_desc.forwardDir.x,
		transform_desc.forwardDir.y,
		transform_desc.forwardDir.z,
		transform_desc.forwardDir.w);

	m_rightDir = DirectX::XMVectorSet(
		transform_desc.rightDir.x,
		transform_desc.rightDir.y,
		transform_desc.rightDir.z,
		transform_desc.rightDir.w);

	m_upDir = DirectX::XMVectorSet(
		transform_desc.upDir.x,
		transform_desc.upDir.y,
		transform_desc.upDir.z,
		transform_desc.upDir.w);

	m_defaultForwardDir = DirectX::XMVectorSet(
		0.0f,
		0.0f,
		1.0f,
		0.0f);

	m_defaultRightDir = DirectX::XMVectorSet(
		1.0f,
		0.0f,
		0.0f,
		0.0f);

	m_defaultUpDir = DirectX::XMVectorSet(
		0.0f,
		1.0f,
		0.0f,
		0.0f);

	RotateEulerAngles(transform_desc.rotation.x, transform_desc.rotation.y, transform_desc.rotation.z);
	Update();

	m_viewMatrix = DirectX::XMMatrixLookAtLH(
		m_position,
		DirectX::XMVectorAdd(m_position, m_forwardDir),
		m_upDir);
}

DirectX::XMMATRIX Transform::GetWorldMatrix()
{
	return m_worldMatrix;
}

DirectX::XMVECTOR Transform::GetPosition()
{
	return m_position;
}

DirectX::XMVECTOR Transform::GetRotation()
{
	return m_rotation;
}

DirectX::XMVECTOR Transform::GetScale()
{
	return m_scale;
}

DirectX::XMVECTOR Transform::GetForwardDir()
{
	return m_forwardDir;
}

DirectX::XMVECTOR Transform::GetRightDir()
{
	return m_rightDir;
}

DirectX::XMVECTOR Transform::GetUpDir()
{
	return m_upDir;
}

DirectX::XMMATRIX Transform::GetViewMatrix()
{
	return m_viewMatrix;
}

void Transform::Update()
{
	m_worldMatrix = DirectX::XMMatrixIdentity();
	m_scalingMatrix = DirectX::XMMatrixScalingFromVector(m_scale);
	m_worldMatrix = DirectX::XMMatrixMultiply(m_worldMatrix, m_scalingMatrix);
	m_rotationMatrix = DirectX::XMMatrixRotationRollPitchYawFromVector(m_rotation);
	m_worldMatrix = DirectX::XMMatrixMultiply(m_worldMatrix, m_rotationMatrix);
	m_translatioMatrix = DirectX::XMMatrixTranslationFromVector(m_position);
	m_worldMatrix = DirectX::XMMatrixMultiply(m_worldMatrix, m_translatioMatrix);

	m_forwardDir = DirectX::XMVector3Normalize(DirectX::XMVector3Transform(m_defaultForwardDir, m_rotationMatrix));
	m_rightDir = DirectX::XMVector3Normalize(DirectX::XMVector3Cross(m_forwardDir, m_defaultUpDir));
	m_upDir = DirectX::XMVector3Normalize(DirectX::XMVector3Cross(m_rightDir, m_forwardDir));

	m_viewMatrix = DirectX::XMMatrixLookAtLH(
		m_position,
		DirectX::XMVectorAdd(m_position, m_forwardDir),
		m_upDir);

	//Currently incorrect X-Axis alignment after rotation
	using namespace DirectX;
	m_rightDir *= -1.0f;
}

void Transform::Translate(DirectX::XMVECTOR translation)
{
	m_position = DirectX::XMVectorAdd(m_position, translation);
}

//Arguments passed in degrees
void Transform::RotateEulerAngles(float pitch, float head, float roll)
{	
	m_rotationDynamic = DirectX::XMFLOAT4(
		pitch + m_rotationDynamic.x,
		head + m_rotationDynamic.y,
		roll + m_rotationDynamic.z,
		0.0f);

	if (m_rotationDynamic.x > 360.0f) {
		m_rotationDynamic.x -= 360.0f;
	}

	if (m_rotationDynamic.x < -360.0f) {
		m_rotationDynamic.x += 360.0f;
	}

	if (m_rotationDynamic.y > 360.0f) {
		m_rotationDynamic.y -= 360.0f;
	}

	if (m_rotationDynamic.y < -360.0f) {
		m_rotationDynamic.y += 360.0f;
	}

	if (m_rotationDynamic.z > 360.0f) {
		m_rotationDynamic.z -= 360.0f;
	}

	if (m_rotationDynamic.z < -360.0f) {
		m_rotationDynamic.z += 360.0f;
	}

	m_rotation = DirectX::XMVectorSet(
		DirectX::XMConvertToRadians(m_rotationDynamic.x),
		DirectX::XMConvertToRadians(m_rotationDynamic.y),
		DirectX::XMConvertToRadians(m_rotationDynamic.z),
			0.0f);
}

void Transform::Scale(DirectX::XMVECTOR scale)
{
	m_scale = scale;
}