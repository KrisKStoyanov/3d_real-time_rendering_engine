#pragma once
#include "Material.h"

struct MESH_DESC 
{
	VERTEX_BUFFER_DESC vertex_buffer_desc;
	INDEX_BUFFER_DESC index_buffer_desc;
};

class Mesh 
{
public:
	static Mesh* Create(GraphicsContext& graphicsContext, MESH_DESC& mesh_desc, MATERIAL_DESC& mat_desc);
	void Shutdown();

	inline Buffer* GetVertexBuffer() { return m_pVertexBuffer; }
	inline Buffer* GetIndexBuffer() { return m_pIndexBuffer; }

	Material* GetMaterial();
private:
	Mesh(GraphicsContext& graphicsContext, MESH_DESC& mesh_desc, MATERIAL_DESC& mat_desc);

	Buffer* m_pVertexBuffer;
	Buffer* m_pIndexBuffer;
	 
	Material* m_pMaterial;
};
