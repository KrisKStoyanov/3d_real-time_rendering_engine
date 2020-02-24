#include "Model.h"

Model* Model::Create(D3D11Context& context, MESH_DESC& mesh_desc, VertexType vertexType)
{
	return new Model(context, mesh_desc, vertexType);
}

Model::Model(D3D11Context& graphicsContext, MESH_DESC& mesh_desc, VertexType vertexType) :
	m_pMesh(nullptr), m_vertexType(vertexType)
{
	m_pMesh = Mesh::Create(graphicsContext, mesh_desc, vertexType);
}

void Model::Shutdown()
{
	SAFE_SHUTDOWN(m_pMesh);
}

Mesh* Model::GetMesh()
{
	return m_pMesh;
}

VertexType Model::GetVertexType()
{
	return m_vertexType;
}