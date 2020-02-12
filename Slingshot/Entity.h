#pragma once
#include "Transform.h"
#include "Renderer.h"

class Entity {

public:
	Entity(TRANSFORM_DESC* transform_desc = nullptr);
	~Entity();

	void Shutdown();

	bool SetModel(Renderer* renderer, MODEL_DESC* model_desc);
	void UnsetModel();

	Transform* GetTransform();
	Model* GetModel();
private:
	Transform* m_pTransform;
	Model* m_pModel;
};