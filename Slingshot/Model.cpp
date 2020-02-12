#include "Model.h"

Model* Model::Create(D3D11Context* context, MODEL_DESC* model_desc)
{
	return new Model(context, model_desc);
}

Model::Model(D3D11Context* graphicsContext, MODEL_DESC* model_desc) : m_pMesh(nullptr)
{
	if ((m_pMesh = Mesh::Create(graphicsContext, model_desc->mesh_desc)) != nullptr) {
		m_pMesh->SetGraphicsProps(graphicsContext, model_desc->shader_desc);
	}
}

void Model::Render(D3D11Context* graphicsContext)
{
	m_pMesh->Render(graphicsContext);
}

void Model::Shutdown()
{
}
