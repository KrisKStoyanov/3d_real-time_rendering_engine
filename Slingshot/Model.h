#pragma once
#include "Mesh.h"

class Model {
public:
	static Model* Create(D3D11Context& graphicsContext, MESH_DESC& mesh_desc, VertexType vertexType);
	void Shutdown();

	Mesh* GetMesh();
	VertexType GetVertexType();
private:
	Model(D3D11Context& graphicsContext, MESH_DESC& mesh_desc, VertexType vertexType);
	Mesh* m_pMesh;
	VertexType m_vertexType;
};

