#pragma once
#include "D3D11Context.h"
#include "Material.h"

struct MESH_DESC {
	Vertex* vertexCollection;
	unsigned int vertexCount;
	unsigned int* indexCollection;
	unsigned int indexCount;
	D3D11_PRIMITIVE_TOPOLOGY topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
};

class Mesh {
public:
	static Mesh* Create(D3D11Context& graphicsContext, MESH_DESC& mesh_desc, MATERIAL_DESC& mat_desc);
	void Shutdown();

	const Microsoft::WRL::ComPtr<ID3D11Buffer> GetVertexBuffer();
	const Microsoft::WRL::ComPtr<ID3D11Buffer> GetIndexBuffer();

	unsigned int GetVertexCount();
	unsigned int GetIndexCount();

	unsigned int* GetVertexBufferStride();
	unsigned int* GetVertexBufferOffset();

	D3D11_PRIMITIVE_TOPOLOGY GetTopology();
	Material* GetMaterial();
private:
	Mesh(D3D11Context& graphicsContext, MESH_DESC& mesh_desc, MATERIAL_DESC& mat_desc);

	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pVBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pIBuffer;

	unsigned int m_vertexCount;
	unsigned int m_indexCount;

	unsigned int m_VBufferStride;
	unsigned int m_VBufferOffset;

	D3D11_PRIMITIVE_TOPOLOGY m_topology;
	 
	Material* m_pMaterial;
};
