#pragma once
#include "GraphicsProps.h"

struct MESH_DESC {
	VertexType vertexType;
	Vertex* vertexCollection;
	unsigned int vertexCount;
	unsigned int* indexCollection;
	unsigned int indexCount;
	D3D11_PRIMITIVE_TOPOLOGY topology;
	MESH_DESC(
		VertexType _vertexType, D3D11_PRIMITIVE_TOPOLOGY _topology,
		Vertex* _vertexCollection, unsigned int _vertexCount,
		unsigned int* _indexCollection, unsigned int _indexCount) :
		vertexType(_vertexType), topology(_topology),
		vertexCollection(_vertexCollection), vertexCount(_vertexCount), 
		indexCollection(_indexCollection), indexCount(_indexCount)
	{}
};

class Mesh {
public:
	static Mesh* Create(D3D11Context* graphicsContext, MESH_DESC* mesh_desc);
	void Shutdown();

	void SetGraphicsProps(D3D11Context* graphicsContext, SHADER_DESC* shader_desc, VertexType vertexType);
	void Render(D3D11Context* graphicsContext);

	int GetVertexCount();
	int GetIndexCount();
private:
	Mesh(D3D11Context* graphicsContext, MESH_DESC* mesh_desc);
	GraphicsProps* m_pGraphicsProps;

	ID3D11Buffer* m_pVBuffer;
	ID3D11Buffer* m_pIBuffer;

	unsigned int m_vertexCount;
	unsigned int m_indexCount;

	unsigned int m_VBufferStride;
	unsigned int m_VBufferOffset;

	D3D11_PRIMITIVE_TOPOLOGY m_topology;
};
