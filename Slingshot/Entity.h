#pragma once
#include "Model.h"
#include "Camera.h"
#include "Light.h"

class Entity {

public:
	Entity();

	void Shutdown();

	void SetTransform(TRANSFORM_DESC& transform_desc);
	void SetModel(GraphicsContext& graphicsContext, MESH_DESC& mesh_desc, MATERIAL_DESC& mat_desc);
	void SetCamera(CAMERA_DESC& camera_desc);
	void SetLight(LIGHT_DESC& light_desc);

	Transform* GetTransform();
	Model* GetModel();
	Camera* GetCamera();
	Light* GetLight();

private:
	Transform* m_pTransform;
	Model* m_pModel;
	Camera* m_pCamera;
	Light* m_pLight;
};