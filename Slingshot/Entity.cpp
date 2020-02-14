#include "Entity.h"

Entity::Entity(TRANSFORM_DESC* transform_desc) : m_pTransform(nullptr), m_pCamera(nullptr), m_pModel(nullptr)
{
	transform_desc != nullptr ? m_pTransform = Transform::Create(transform_desc) : m_pTransform = Transform::Create(&TRANSFORM_DESC());
}

void Entity::Shutdown()
{
	SAFE_SHUTDOWN(m_pModel);
	SAFE_DELETE(m_pTransform);
}

bool Entity::SetTransform(TRANSFORM_DESC* transform_desc)
{
	return ((m_pTransform = Transform::Create(transform_desc)) != nullptr);
}

bool Entity::SetModel(D3D11Context* graphicsContext, MODEL_DESC* model_desc)
{
	return ((m_pModel = Model::Create(graphicsContext, model_desc)) != nullptr);
}

bool Entity::SetCamera(CAMERA_DESC* camera_desc)
{
	return ((m_pCamera = Camera::Create(camera_desc, m_pTransform)) != nullptr);
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
