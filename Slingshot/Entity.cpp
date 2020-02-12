#include "Entity.h"

Entity::Entity(TRANSFORM_DESC* transform_desc) : m_pTransform(nullptr), m_pModel(nullptr)
{
	if (transform_desc) {
		m_pTransform = new Transform(transform_desc);
	}
	else {
		m_pTransform = new Transform();
	}
}

Entity::~Entity()
{
}

void Entity::Shutdown()
{

}

bool Entity::SetModel(Renderer* renderer, MODEL_DESC* model_desc)
{
	return ((m_pModel = Model::Create(renderer->GetGraphicsContext(), model_desc)) != nullptr);
}

void Entity::UnsetModel()
{

}

Transform* Entity::GetTransform()
{
	return m_pTransform;
}

Model* Entity::GetModel()
{
	return m_pModel;
}
