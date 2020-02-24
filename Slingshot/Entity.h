#pragma once
#include "Model.h"
#include "Camera.h"

class Entity {

public:
	Entity();

	void Shutdown();

	void SetTransform(TRANSFORM_DESC& transform_desc);
	void SetModel(D3D11Context& graphicsContext, MESH_DESC& mesh_desc, VertexType vertexType);
	void SetCamera(CAMERA_DESC& camera_desc);

	Transform* GetTransform();
	Model* GetModel();
	Camera* GetCamera();
private:
	Transform* m_pTransform;
	Model* m_pModel;
	Camera* m_pCamera;
};