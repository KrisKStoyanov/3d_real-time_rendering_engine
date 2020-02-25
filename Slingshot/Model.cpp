#include "Model.h"

Model* Model::Create(D3D11Context& context, MESH_DESC& mesh_desc, ShadingModel shadingModel)
{
	return new Model(context, mesh_desc, shadingModel);
}

Model::Model(D3D11Context& graphicsContext, MESH_DESC& mesh_desc, ShadingModel shadingModel) :
	m_pMesh(nullptr), m_shadingModel(shadingModel)
{
	m_pMesh = Mesh::Create(graphicsContext, mesh_desc, shadingModel);
}

void Model::Shutdown()
{
	SAFE_SHUTDOWN(m_pMesh);
}

Mesh* Model::GetMesh()
{
	return m_pMesh;
}

ShadingModel Model::GetShadingModel()
{
	return m_shadingModel;
}