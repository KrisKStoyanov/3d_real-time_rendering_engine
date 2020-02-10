#pragma once
#include "Transform.h"
#include "GraphicsProps.h"

class Entity {

public:
	Transform* GetTransform();
	GraphicsProps* GetGraphicsProps();
private:
	Transform* m_pTransform;
	GraphicsProps* m_gProps;
};