#include "Entity.h"

Transform* Entity::GetTransform()
{
	return m_pTransform;
}

GraphicsProps* Entity::GetGraphicsProps()
{
	return m_gProps;
}
