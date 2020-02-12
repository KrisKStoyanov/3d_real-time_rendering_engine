#pragma once
#include "GraphicsProps.h"

struct MESH_DESC {
	ColorShaderVertex* vertexCollection;
	unsigned int vertexCount;
	unsigned int* indexCollection;
	unsigned int indexCount;
	MESH_DESC(
		ColorShaderVertex* _vertexCollection, unsigned int _vertexCount, 
		unsigned int* _indexCollection, unsigned int _indexCount) : 
		vertexCollection(_vertexCollection), vertexCount(_vertexCount), 
		indexCollection(_indexCollection), indexCount(_indexCount) {}
};

class Mesh {
public:
	static Mesh* Create(D3D11Context* graphicsContext, MESH_DESC* mesh_desc);
	void Shutdown();

	void SetGraphicsProps(D3D11Context* graphicsContext, SHADER_DESC* shader_desc);
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
};
