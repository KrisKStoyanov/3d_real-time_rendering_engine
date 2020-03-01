#include "Model.h"

Model* Model::Create(D3D11Context& context, MESH_DESC& mesh_desc, MATERIAL_DESC& mat_desc)
{
	return new Model(context, mesh_desc, mat_desc);
}

Model::Model(D3D11Context& graphicsContext, MESH_DESC& mesh_desc, MATERIAL_DESC& mat_desc) :
	m_pMesh(nullptr)
{
	m_pMesh = Mesh::Create(graphicsContext, mesh_desc, mat_desc);
}

void Model::Shutdown()
{
	SAFE_SHUTDOWN(m_pMesh);
}

Mesh* Model::GetMesh()
{
	return m_pMesh;
}