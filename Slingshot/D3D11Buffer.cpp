#include "D3D11Buffer.h"

D3D11VertexBuffer* D3D11VertexBuffer::Create(ID3D11Device& device, VERTEX_BUFFER_DESC desc)
{
	return new D3D11VertexBuffer(device, desc);
}

void D3D11VertexBuffer::Destroy()
{
	SAFE_RELEASE(m_pBuffer);
}

void D3D11VertexBuffer::Bind(ID3D11DeviceContext& context)
{
	context.IASetVertexBuffers(0, 1,
		&m_pBuffer,
		&m_stride,
		&m_offset);
	context.IASetPrimitiveTopology(m_topology);
}


D3D11VertexBuffer::D3D11VertexBuffer(ID3D11Device& device, VERTEX_BUFFER_DESC desc) :
	m_stride(desc.stride), m_offset(desc.offset), 
	m_vertexCount(desc.vertexCount), m_pBuffer(nullptr)
{
	D3D11_BUFFER_DESC vertexBufferDesc;
	ZeroMemory(&vertexBufferDesc, sizeof(vertexBufferDesc));
	vertexBufferDesc.Usage = D3D11_USAGE::D3D11_USAGE_DEFAULT;
	vertexBufferDesc.ByteWidth = desc.stride * desc.vertexCount;
	vertexBufferDesc.BindFlags = D3D11_BIND_FLAG::D3D11_BIND_VERTEX_BUFFER;
	vertexBufferDesc.CPUAccessFlags = 0;
	vertexBufferDesc.MiscFlags = 0;
	vertexBufferDesc.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vertexData;
	ZeroMemory(&vertexData, sizeof(vertexData));
	vertexData.pSysMem = desc.vertexCollection;
	vertexData.SysMemPitch = 0;
	vertexData.SysMemSlicePitch = 0;

	DX::ThrowIfFailed(device.CreateBuffer(
		&vertexBufferDesc, &vertexData, &m_pBuffer));

	switch (desc.topology)
	{
	case Topology::TRIANGLESTRIP:
	{
		m_topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
	}
	break;
	case Topology::TRIANGLELIST:
	{
		m_topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	}
	break;
	case Topology::LINESTRIP:
	{
		m_topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP;
	}
	break;
	case Topology::LINELIST:
	{
		m_topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_LINELIST;
	}
	break;
	case Topology::POINTLIST:
	{
		m_topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_POINTLIST;
	}
	default: //PATCHLIST pending implementation
	{
		m_topology = D3D11_PRIMITIVE_TOPOLOGY::D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
	}
	break;
	}

	SAFE_DELETE_ARRAY(desc.vertexCollection);
}

D3D11IndexBuffer* D3D11IndexBuffer::Create(ID3D11Device& device, INDEX_BUFFER_DESC desc)
{
	return new D3D11IndexBuffer(device, desc);
}

void D3D11IndexBuffer::Destroy()
{
	SAFE_RELEASE(m_pBuffer);
}

void D3D11IndexBuffer::Bind(ID3D11DeviceContext& deviceContext)
{
	deviceContext.IASetIndexBuffer(m_pBuffer, DXGI_FORMAT_R32_UINT, 0);
}

D3D11IndexBuffer::D3D11IndexBuffer(ID3D11Device& device, INDEX_BUFFER_DESC desc) :
	m_indexCount(desc.indexCount), m_pBuffer(nullptr)
{
	D3D11_BUFFER_DESC indexBufferDesc;
	ZeroMemory(&indexBufferDesc, sizeof(indexBufferDesc));
	indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
	indexBufferDesc.ByteWidth = sizeof(unsigned int) * desc.indexCount;
	indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	indexBufferDesc.CPUAccessFlags = 0;
	indexBufferDesc.MiscFlags = 0;
	indexBufferDesc.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA indexData;
	ZeroMemory(&indexData, sizeof(indexData));
	indexData.pSysMem = desc.indexCollection;
	indexData.SysMemPitch = 0;
	indexData.SysMemSlicePitch = 0;

	DX::ThrowIfFailed(device.CreateBuffer(
		&indexBufferDesc, &indexData, &m_pBuffer));

	SAFE_DELETE_ARRAY(desc.indexCollection);
}

D3D11ConstantBuffer* D3D11ConstantBuffer::Create(ID3D11Device& device, CONSTANT_BUFFER_DESC desc)
{
	return new D3D11ConstantBuffer(device, desc);
}

D3D11ConstantBuffer::D3D11ConstantBuffer(ID3D11Device& device, CONSTANT_BUFFER_DESC desc)
	: m_shaderType(desc.shaderType)
{
	//Ensure size is valid (multiple of 16)
	desc.cbufferSize = ((desc.cbufferSize - 1) | 15) + 1;

	m_pData = desc.cbufferData;

	D3D11_BUFFER_DESC vs_cb_desc;
	ZeroMemory(&vs_cb_desc, sizeof(vs_cb_desc));
	vs_cb_desc.Usage = D3D11_USAGE_DEFAULT;
	vs_cb_desc.ByteWidth = desc.cbufferSize; 
	vs_cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	vs_cb_desc.CPUAccessFlags = 0;
	vs_cb_desc.MiscFlags = 0;
	vs_cb_desc.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vs_cb_data;
	vs_cb_data.pSysMem = &m_pData;
	vs_cb_data.SysMemPitch = 0;
	vs_cb_data.SysMemSlicePitch = 0;

	DX::ThrowIfFailed(device.CreateBuffer(
		&vs_cb_desc, &vs_cb_data, &m_pBuffer));

	SAFE_DELETE(desc.cbufferData);
}

void D3D11ConstantBuffer::Destroy()
{
	SAFE_RELEASE(m_pBuffer);
	SAFE_DELETE(m_pData);
}

void D3D11ConstantBuffer::Update(CBufferData& data)
{
	m_pData = &data;
}

void D3D11ConstantBuffer::Bind(ID3D11DeviceContext& deviceContext)
{
	deviceContext.UpdateSubresource(m_pBuffer, 0, nullptr, m_pData, 0, 0);
	switch (m_shaderType)
	{
	case ShaderType::VERTEX_SHADER:
	{
		deviceContext.VSSetConstantBuffers(0, 1, &m_pBuffer);
	}
	break;
	case ShaderType::PIXEL_SHADER:
	{
		deviceContext.PSSetConstantBuffers(0, 1, &m_pBuffer);
	}
	break;
	}
}
