#pragma once
#include "Model.h"
#include "Camera.h"

class Entity {

public:
	Entity();

	void Shutdown();

	bool SetTransform(TRANSFORM_DESC& transform_desc);
	bool SetModel(D3D11Context& graphicsContext, MODEL_DESC& model_desc);
	bool SetCamera(CAMERA_DESC& camera_desc);

	Transform* GetTransform();
	Model* GetModel();
	Camera* GetCamera();
private:
	Transform* m_pTransform;
	Model* m_pModel;
	Camera* m_pCamera;
};