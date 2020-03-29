#include "Mesh.h"

Mesh* Mesh::Create(GraphicsContext& graphicsContext, MESH_DESC& mesh_desc, MATERIAL_DESC& mat_desc)
{
	return new Mesh(graphicsContext, mesh_desc, mat_desc);
}

void Mesh::Shutdown()
{
	//m_pVBuffer->Release();
	//m_pIBuffer->Release();
	//m_pVSCB->Release();
	//SAFE_RELEASE(m_pVBuffer);
	//SAFE_RELEASE(m_pIBuffer);
	//SAFE_RELEASE(m_pVSCB);
	SAFE_DELETE(m_pMaterial);
}

Mesh::Mesh(GraphicsContext& graphicsContext, MESH_DESC& mesh_desc, MATERIAL_DESC& mat_desc) :
	m_pVertexBuffer(nullptr), m_pIndexBuffer(nullptr), m_pMaterial(nullptr)
{
	m_pMaterial = Material::Create(mat_desc);

	switch (mat_desc.shadingModel)
	{
	case ShadingModel::GoochShading:
	{
		mesh_desc.vertex_buffer_desc.stride = sizeof(GoochShadingVertex);
		mesh_desc.vertex_buffer_desc.offset = 0;
	}
	break;
	case ShadingModel::OrenNayarShading:
	{
		mesh_desc.vertex_buffer_desc.stride = sizeof(OrenNayarVertex);
		mesh_desc.vertex_buffer_desc.offset = 0;
	}
	break;
	}

	m_pVertexBuffer = graphicsContext.CreateVertexBuffer(mesh_desc.vertex_buffer_desc);
	m_pIndexBuffer = graphicsContext.CreateIndexBuffer(mesh_desc.index_buffer_desc);
}

Material* Mesh::GetMaterial()
{
	return m_pMaterial;
}
