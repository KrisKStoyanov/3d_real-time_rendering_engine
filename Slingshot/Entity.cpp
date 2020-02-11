#include "Entity.h"

Entity::Entity()
{
}

Entity::~Entity()
{
}

void Entity::AttachGraphicsProps(GraphicsProps gProps)
{
	m_gProps = gProps;
}

void Entity::DetachGraphicsProps()
{
	m_gProps.Clear();
}

Transform Entity::GetTransform()
{
	return m_transform;
}

GraphicsProps Entity::GetGraphicsProps()
{
	return m_gProps;
}
