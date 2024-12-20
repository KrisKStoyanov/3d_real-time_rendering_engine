#pragma once
#include "Buffer.h"

#include <d3d11.h>
#include "d3d11_1.h"

class D3D11VertexBuffer : public Buffer
{
public:
	static D3D11VertexBuffer* Create(
		ID3D11Device& device, 
		VERTEX_BUFFER_DESC desc);
	void Destroy() override;
	void Bind(ID3D11DeviceContext& deviceContext);

	unsigned int GetElementCount() override;
private:
	D3D11VertexBuffer(
		ID3D11Device& device,
		VERTEX_BUFFER_DESC desc);

	ID3D11Buffer* m_pBuffer;
	unsigned int m_stride;
	unsigned int m_offset;
	D3D11_PRIMITIVE_TOPOLOGY m_topology;
	unsigned int m_vertexCount;
};

class D3D11IndexBuffer : public Buffer
{
public:
	static D3D11IndexBuffer* Create(
		ID3D11Device& device,
		INDEX_BUFFER_DESC desc);
	void Destroy() override;
	void Bind(ID3D11DeviceContext& deviceContext);

	unsigned int GetElementCount() override;
private:
	D3D11IndexBuffer(
		ID3D11Device& device,
		INDEX_BUFFER_DESC desc);

	ID3D11Buffer* m_pBuffer;
	unsigned int m_indexCount;
};

class D3D11ConstantBuffer : public Buffer
{
public:
	static D3D11ConstantBuffer* Create(
		ID3D11Device& device,
		CONSTANT_BUFFER_DESC desc);
	void Destroy() override; 
	void Bind(ID3D11DeviceContext& deviceContext, void* data);

	unsigned int GetElementCount() override;
private:
	D3D11ConstantBuffer(
		ID3D11Device& device,
		CONSTANT_BUFFER_DESC desc);
	ID3D11Buffer* m_pBuffer;
	ShaderType m_shaderType;
	unsigned int m_registerSlot;
};

class D3D11StructuredBuffer : public Buffer
{
public:
	static D3D11StructuredBuffer* Create(
		ID3D11Device& device,
		STRUCTURED_BUFFER_DESC desc);
	void Destroy() override;

	unsigned int GetElementCount() override;

	inline ID3D11Buffer* GetBuffer() { return m_pBuffer; }
private:
	D3D11StructuredBuffer(
		ID3D11Device& device,
		STRUCTURED_BUFFER_DESC desc);
	ID3D11Buffer* m_pBuffer;

	unsigned int m_registerSlot;

	unsigned int m_numStructs;
	unsigned int m_numElementsPerStruct;
};