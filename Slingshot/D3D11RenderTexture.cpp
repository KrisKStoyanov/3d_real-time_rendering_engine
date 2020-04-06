#include "D3D11RenderTexture.h"

D3D11RenderTexture* D3D11RenderTexture::Create(ID3D11Device& device, RENDER_TEXTURE_DESC& desc)
{
	return new D3D11RenderTexture(device, desc);
}

D3D11RenderTexture::D3D11RenderTexture(ID3D11Device& device, RENDER_TEXTURE_DESC& desc) :
	m_pRenderTargetView(nullptr),
	m_pShaderResourceView(nullptr),
	m_clearColor(), 
	m_registerSlot(desc.registerSlot),
	m_shaderType(desc.shaderType)
{
	m_clearColor[0] = 0.0f;
	m_clearColor[1] = 0.0f;
	m_clearColor[2] = 0.0f;
	m_clearColor[3] = 1.0f;

	ID3D11Texture2D* renderTexture;

	D3D11_TEXTURE2D_DESC textureDesc;
	ZeroMemory(&textureDesc, sizeof(textureDesc));
	textureDesc.Width = desc.textureWidth;
	textureDesc.Height = desc.textureHeight;
	textureDesc.MipLevels = 1;
	textureDesc.ArraySize = 1;
	textureDesc.Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
	textureDesc.SampleDesc.Count = 1;
	textureDesc.Usage = D3D11_USAGE_DEFAULT;
	textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	textureDesc.CPUAccessFlags = 0;
	textureDesc.MiscFlags = 0;
	DX::ThrowIfFailed(
		device.CreateTexture2D(&textureDesc, nullptr, &renderTexture));

	D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
	rtvDesc.Format = textureDesc.Format;
	rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
	rtvDesc.Texture2D.MipSlice = 0;
	DX::ThrowIfFailed(
		device.CreateRenderTargetView(renderTexture, &rtvDesc, &m_pRenderTargetView));

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	srvDesc.Format = textureDesc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = 1;
	DX::ThrowIfFailed(
		device.CreateShaderResourceView(renderTexture, &srvDesc, &m_pShaderResourceView));

	SAFE_RELEASE(renderTexture);
}

void D3D11RenderTexture::Shutdown()
{
	SAFE_RELEASE(m_pRenderTargetView);
	SAFE_RELEASE(m_pShaderResourceView);
}

void D3D11RenderTexture::SetRenderTarget(ID3D11DeviceContext& deviceContext, ID3D11DepthStencilView& depthStencilView)
{
	deviceContext.OMSetRenderTargets(1, &m_pRenderTargetView, &depthStencilView);
}

void D3D11RenderTexture::ClearRenderTarget(ID3D11DeviceContext& deviceContext, ID3D11DepthStencilView& depthStencilView)
{
	deviceContext.ClearRenderTargetView(m_pRenderTargetView, m_clearColor);
	deviceContext.ClearDepthStencilView(&depthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0); 
}

void D3D11RenderTexture::SetShaderResource(ID3D11DeviceContext& deviceContext)
{
	switch (m_shaderType)
	{
	case ShaderType::PIXEL_SHADER:
	{
		deviceContext.PSSetShaderResources(m_registerSlot, 1, &m_pShaderResourceView);
	}
	break;
	case ShaderType::VERTEX_SHADER:
	{
		deviceContext.VSSetShaderResources(m_registerSlot, 1, &m_pShaderResourceView);
	}
	break;
	}
}

void D3D11RenderTexture::UnsetShaderResource(ID3D11DeviceContext& deviceContext)
{
	ID3D11ShaderResourceView* nullSRV[1] = { nullptr };
	switch (m_shaderType)
	{
	case ShaderType::PIXEL_SHADER:
	{
		deviceContext.PSSetShaderResources(m_registerSlot, 1, nullSRV);
	}
	break;
	case ShaderType::VERTEX_SHADER:
	{
		deviceContext.VSSetShaderResources(m_registerSlot, 1, nullSRV);
	}
	break;
	}
}
