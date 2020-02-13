#include "Entity.h"

Entity::Entity(TRANSFORM_DESC* transform_desc) : m_pTransform(nullptr), m_pModel(nullptr)
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

bool Entity::SetModel(Renderer* renderer, MODEL_DESC* model_desc)
{
	return ((m_pModel = Model::Create(renderer->GetGraphicsContext(), model_desc)) != nullptr);
}

Transform* Entity::GetTransform()
{
	return m_pTransform;
}

Model* Entity::GetModel()
{
	return m_pModel;
}
