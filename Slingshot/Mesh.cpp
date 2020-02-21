#include "Mesh.h"

Mesh* Mesh::Create(D3D11Context& graphicsContext, MESH_DESC& mesh_desc)
{
	return new Mesh(graphicsContext, mesh_desc);
}

void Mesh::Shutdown()
{
	m_pGraphicsProps->Shutdown();
	SAFE_RELEASE(m_pVBuffer);
	SAFE_RELEASE(m_pIBuffer);
	SAFE_RELEASE(m_pVSCB);
}

Mesh::Mesh(D3D11Context& graphicsContext, MESH_DESC& mesh_desc) :
	m_pGraphicsProps(nullptr),
	m_pVSCB(nullptr), m_pVBuffer(nullptr), m_pIBuffer(nullptr),
	m_vertexCount(mesh_desc.vertexCount), m_indexCount(mesh_desc.indexCount),
	m_VBufferStride(0), m_VBufferOffset(0), m_topology(mesh_desc.topology)
{
	switch (mesh_desc.vertexType)
	{
	case VertexType::ColorShaderVertex:
	{
		m_VBufferStride = sizeof(ColorShaderVertex);
		m_VBufferOffset = 0;
	}
	break;
	default:
	{
		m_VBufferStride = sizeof(ColorShaderVertex);
		m_VBufferOffset = 0;
	}
	break;
	}

	//----------------------------------

	D3D11_BUFFER_DESC vertexBufferDesc;
	ZeroMemory(&vertexBufferDesc, sizeof(vertexBufferDesc));
	vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
	vertexBufferDesc.ByteWidth = m_VBufferStride * mesh_desc.vertexCount;
	vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vertexBufferDesc.CPUAccessFlags = 0;
	vertexBufferDesc.MiscFlags = 0;
	vertexBufferDesc.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vertexData;
	ZeroMemory(&vertexData, sizeof(vertexData));
	vertexData.pSysMem = mesh_desc.vertexCollection;
	vertexData.SysMemPitch = 0;
	vertexData.SysMemSlicePitch = 0;

	DX::ThrowIfFailed(graphicsContext.GetDevice()->CreateBuffer(
		&vertexBufferDesc, &vertexData, &m_pVBuffer));

	//----------------------------------

	D3D11_BUFFER_DESC indexBufferDesc;
	ZeroMemory(&indexBufferDesc, sizeof(indexBufferDesc));
	indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
	indexBufferDesc.ByteWidth = sizeof(*mesh_desc.indexCollection) * mesh_desc.indexCount;
	indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	indexBufferDesc.CPUAccessFlags = 0;
	indexBufferDesc.MiscFlags = 0;
	indexBufferDesc.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA indexData;
	ZeroMemory(&indexData, sizeof(indexData));
	indexData.pSysMem = mesh_desc.indexCollection;
	indexData.SysMemPitch = 0;
	indexData.SysMemSlicePitch = 0;

	DX::ThrowIfFailed(graphicsContext.GetDevice()->CreateBuffer(
		&indexBufferDesc, &indexData, &m_pIBuffer));

	//----------------------------------

	VS_CONSTANT_BUFFER vs_cb;

	D3D11_BUFFER_DESC vs_cb_desc;
	ZeroMemory(&vs_cb_desc, sizeof(vs_cb_desc));
	vs_cb_desc.Usage = D3D11_USAGE_DEFAULT;
	vs_cb_desc.ByteWidth = 64; // sizeof(VS_CONSTANT_BUFFER) = 64 <- constant buffer size must be a multiple of 16 bytes;
	vs_cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	vs_cb_desc.CPUAccessFlags = 0;
	vs_cb_desc.MiscFlags = 0;
	vs_cb_desc.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vs_cb_data;
	vs_cb_data.pSysMem = &vs_cb;
	vs_cb_data.SysMemPitch = 0;
	vs_cb_data.SysMemSlicePitch = 0;

	DX::ThrowIfFailed(graphicsContext.GetDevice()->CreateBuffer(&vs_cb_desc, &vs_cb_data, &m_pVSCB));

	//----------------------------------
}

void Mesh::SetGraphicsProps(D3D11Context& renderer, SHADER_DESC& shader_desc, VertexType vertexType)
{
	m_pGraphicsProps = GraphicsProps::Create(renderer, shader_desc, vertexType);
}

void Mesh::OnFrameRender(D3D11Context& graphicsContext, DirectX::XMMATRIX wvp)
{
	ID3D11DeviceContext* deviceContext = graphicsContext.GetDeviceContext();
	deviceContext->IASetVertexBuffers(0, 1, &m_pVBuffer, &m_VBufferStride, &m_VBufferOffset);
	deviceContext->IASetIndexBuffer(m_pIBuffer, DXGI_FORMAT_R32_UINT, 0);
	deviceContext->IASetPrimitiveTopology(m_topology);

	deviceContext->IASetInputLayout(m_pGraphicsProps->GetInputLayout());
	deviceContext->VSSetShader(m_pGraphicsProps->GetVertexShader(), nullptr, 0);
	deviceContext->PSSetShader(m_pGraphicsProps->GetPixelShader(), nullptr, 0);

	VS_CONSTANT_BUFFER vs_cb(wvp);
	deviceContext->UpdateSubresource(m_pVSCB, 0, nullptr, &vs_cb, 0, 0);
	deviceContext->VSSetConstantBuffers(0, 1, &m_pVSCB);

	deviceContext->DrawIndexed(m_indexCount, 0, 0);
}

int Mesh::GetVertexCount()
{
	return m_vertexCount;
}

int Mesh::GetIndexCount()
{
	return m_indexCount;
}
