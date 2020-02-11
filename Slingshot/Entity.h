#pragma once
#include "Transform.h"
#include "GraphicsProps.h"

class Entity {

public:
	Entity();
	~Entity();

	void AttachGraphicsProps(GraphicsProps gProps);
	void DetachGraphicsProps();

	Transform GetTransform();
	GraphicsProps GetGraphicsProps();
private:
	Transform m_transform;
	GraphicsProps m_gProps;
};