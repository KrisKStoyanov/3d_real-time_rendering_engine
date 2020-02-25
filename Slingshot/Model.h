#pragma once
#include "Mesh.h"

class Model {
public:
	static Model* Create(D3D11Context& graphicsContext, MESH_DESC& mesh_desc, ShadingModel shadingModel);
	void Shutdown();

	Mesh* GetMesh();
	ShadingModel GetShadingModel();
private:
	Model(D3D11Context& graphicsContext, MESH_DESC& mesh_desc, ShadingModel shadingModel);
	Mesh* m_pMesh;
	ShadingModel m_shadingModel;
};

