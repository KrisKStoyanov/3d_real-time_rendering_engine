#include "Model.h"

Model* Model::Create(D3D11Context& context, MODEL_DESC& model_desc)
{
	return new Model(context, model_desc);
}

Model::Model(D3D11Context& graphicsContext, MODEL_DESC& model_desc) : 
	m_pMesh(nullptr), m_pGraphicsProps(nullptr)
{
	m_pMesh = Mesh::Create(graphicsContext, model_desc.mesh_desc);
	m_pGraphicsProps = GraphicsProps::Create(graphicsContext, model_desc.shader_desc, model_desc.mesh_desc.vertexType);
}

void Model::Shutdown()
{
	SAFE_SHUTDOWN(m_pMesh);
	SAFE_SHUTDOWN(m_pGraphicsProps);
}

Mesh* Model::GetMesh()
{
	return m_pMesh;
}

GraphicsProps* Model::GetGraphicsProps()
{
	return m_pGraphicsProps;
}
