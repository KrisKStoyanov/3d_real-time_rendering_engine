#pragma once
#include "D3D11Buffer.h"
#include "Material.h"

struct MESH_DESC 
{
	VERTEX_BUFFER_DESC vertex_buffer_desc;
	INDEX_BUFFER_DESC index_buffer_desc;
};

class Mesh 
{
public:
	static Mesh* Create(D3D11Context& graphicsContext, MESH_DESC& mesh_desc, MATERIAL_DESC& mat_desc);
	void Shutdown();

	inline D3D11VertexBuffer* GetVertexBuffer() { return m_pVertexBuffer; }
	inline D3D11IndexBuffer* GetIndexBuffer() { return m_pIndexBuffer; }

	Material* GetMaterial();
private:
	Mesh(D3D11Context& graphicsContext, MESH_DESC& mesh_desc, MATERIAL_DESC& mat_desc);

	D3D11VertexBuffer* m_pVertexBuffer;
	D3D11IndexBuffer* m_pIndexBuffer;
	 
	Material* m_pMaterial;
};
