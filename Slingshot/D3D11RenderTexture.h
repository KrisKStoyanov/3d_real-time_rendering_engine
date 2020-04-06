#pragma once
#include "D3D11Buffer.h"

struct RENDER_TEXTURE_DESC
{
	int textureWidth;
	int textureHeight;
	ShaderType shaderType;
	unsigned int registerSlot;
};

class D3D11RenderTexture
{
public:
	static D3D11RenderTexture* Create(ID3D11Device& device, RENDER_TEXTURE_DESC& desc);
	void Shutdown();

	void SetRenderTarget(ID3D11DeviceContext& deviceContext, ID3D11DepthStencilView& depthStencilView);
	void ClearRenderTarget(ID3D11DeviceContext& deviceContext, ID3D11DepthStencilView& depthStencilView);
	void SetShaderResource(ID3D11DeviceContext& deviceContext);
	void UnsetShaderResource(ID3D11DeviceContext& deviceContext);
	inline ID3D11ShaderResourceView* const GetShaderResourceView()
	{
		return m_pShaderResourceView;
	}

private:
	D3D11RenderTexture(ID3D11Device& device, RENDER_TEXTURE_DESC& desc);

	float m_clearColor[4];

	ID3D11Texture2D* m_pRenderTargetTexture;
	ID3D11RenderTargetView* m_pRenderTargetView;
	ID3D11ShaderResourceView* m_pShaderResourceView;

	ShaderType m_shaderType;
	unsigned int m_registerSlot;
};

