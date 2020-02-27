#include "Entity.h"

Entity::Entity() : 
	m_pTransform(nullptr), m_pCamera(nullptr), 
	m_pModel(nullptr), m_pLight(nullptr)
{
	TRANSFORM_DESC transform_desc;
	m_pTransform = Transform::Create(transform_desc);
}

void Entity::Shutdown()
{
	SAFE_SHUTDOWN(m_pModel);
	SAFE_DELETE(m_pTransform);
	SAFE_DELETE(m_pCamera);
	SAFE_DELETE(m_pLight);
}

void Entity::SetTransform(TRANSFORM_DESC& transform_desc)
{
	m_pTransform = Transform::Create(transform_desc);
}

void Entity::SetModel(D3D11Context& graphicsContext, MESH_DESC& mesh_desc, ShadingModel shadingModel)
{
	m_pModel = Model::Create(graphicsContext, mesh_desc, shadingModel);
}

void Entity::SetCamera(CAMERA_DESC& camera_desc)
{
	m_pCamera = Camera::Create(camera_desc, *m_pTransform);
}

void Entity::SetLight(LIGHT_DESC& light_desc)
{
	m_pLight = Light::Create(light_desc);
}

Transform* Entity::GetTransform()
{
	return m_pTransform;
}

Model* Entity::GetModel()
{
	return m_pModel;
}

Camera* Entity::GetCamera() 
{
	return m_pCamera;
}

Light* Entity::GetLight()
{
	return m_pLight;
}
