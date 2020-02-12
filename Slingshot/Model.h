#pragma once
#include "Mesh.h"

struct MODEL_DESC {
	MESH_DESC* mesh_desc;
	SHADER_DESC* shader_desc;
	MODEL_DESC(MESH_DESC* _mesh_desc, SHADER_DESC* _shader_desc) :
		mesh_desc(_mesh_desc), shader_desc(_shader_desc) {}
};

class Model {
public:
	static Model* Create(D3D11Context* graphicsContext, MODEL_DESC* model_desc);
	void Render(D3D11Context* graphicsContext);
	void Shutdown();

private:
	Model(D3D11Context* graphicsContext, MODEL_DESC* model_desc);
	Mesh* m_pMesh;
};

