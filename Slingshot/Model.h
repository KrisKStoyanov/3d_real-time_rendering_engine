#pragma once
#include "Mesh.h"

struct MODEL_DESC {
	MESH_DESC mesh_desc;
	SHADER_DESC shader_desc;
};

class Model {
public:
	static Model* Create(D3D11Context& graphicsContext, MODEL_DESC& model_desc);
	void OnFrameRender(D3D11Context& graphicsContext, DirectX::XMMATRIX = DirectX::XMMatrixIdentity());
	void Shutdown();

	Mesh* GetMesh();
private:
	Model(D3D11Context& graphicsContext, MODEL_DESC& model_desc);
	Mesh* m_pMesh;
};

