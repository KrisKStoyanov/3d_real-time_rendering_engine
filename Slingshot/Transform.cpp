#include "Transform.h"

Transform::Transform()
{
}

Transform::Transform(TRANSFORM_DESC* transform_desc)
{
	m_pDesc = transform_desc;
}

Transform::~Transform()
{
}

void Transform::Shutdown()
{
}
