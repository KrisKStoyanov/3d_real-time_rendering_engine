#include "Transform.h"

Transform* Transform::Create(TRANSFORM_DESC* transform_desc)
{
	return new Transform(transform_desc);
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

DirectX::XMVECTOR Transform::GetUpDir()
{
	return m_upDir;
}

Transform::Transform(TRANSFORM_DESC* transform_desc) : 
	m_worldMatrix(), m_position(), m_rotation(), m_scale(), m_forwardDir(), m_upDir()
{
	m_worldMatrix = DirectX::XMMatrixIdentity();

	m_position = DirectX::XMVectorSet(
		transform_desc->position.x, 
		transform_desc->position.y, 
		transform_desc->position.z, 
		transform_desc->position.w);

	m_rotation = DirectX::XMVectorSet(
		transform_desc->rotation.x,
		transform_desc->rotation.y,
		transform_desc->rotation.z,
		transform_desc->rotation.w);

	m_scale = DirectX::XMVectorSet(
		transform_desc->scale.x,
		transform_desc->scale.y,
		transform_desc->scale.z,
		transform_desc->scale.w);

	m_forwardDir = DirectX::XMVectorSet(
		transform_desc->forwardDir.x,
		transform_desc->forwardDir.y,
		transform_desc->forwardDir.z,
		transform_desc->forwardDir.w);

	m_upDir = DirectX::XMVectorSet(
		transform_desc->upDir.x,
		transform_desc->upDir.y,
		transform_desc->upDir.z,
		transform_desc->upDir.w);
}