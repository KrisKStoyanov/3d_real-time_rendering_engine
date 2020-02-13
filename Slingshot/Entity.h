#pragma once
#include "Renderer.h"

class Entity {

public:
	Entity(TRANSFORM_DESC* transform_desc = nullptr);

	void Shutdown();

	bool SetTransform(TRANSFORM_DESC* transform_desc);
	bool SetModel(Renderer* renderer, MODEL_DESC* model_desc);

	Transform* GetTransform();
	Model* GetModel();
private:
	Transform* m_pTransform;
	Model* m_pModel;
};