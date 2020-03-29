#pragma once
#include "Mesh.h"

class Model {
public:
	static Model* Create(GraphicsContext& graphicsContext, MESH_DESC& mesh_desc, MATERIAL_DESC& mat_desc);
	void Shutdown();

	Mesh* GetMesh();
private:
	Model(GraphicsContext& graphicsContext, MESH_DESC& mesh_desc, MATERIAL_DESC& mat_desc);
	Mesh* m_pMesh;
};

