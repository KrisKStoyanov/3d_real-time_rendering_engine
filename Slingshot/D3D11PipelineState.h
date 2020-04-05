#pragma once
#include "PipelineState.h"
#include "D3D11RenderTexture.h"

struct PerFrameDataVS
{
	DirectX::XMMATRIX cameraViewMatrix;
	DirectX::XMMATRIX cameraProjMatrix;
	DirectX::XMMATRIX lightViewMatrix;
	DirectX::XMMATRIX lightProjMatrix;
	DirectX::XMVECTOR lightPos;
};

struct PerDrawCallDataVS
{
	DirectX::XMMATRIX worldMatrix;
};

struct PerFrameDataPS 
{
	DirectX::XMFLOAT4 ambientColor; 
	DirectX::XMFLOAT4 diffuseColor; 
};

struct PerDrawCallDataPS 
{
	DirectX::XMFLOAT4 surfaceColor;
};

class D3D11PipelineState : public PipelineState
{
public:
	static D3D11PipelineState* Create(
		ID3D11Device& device, 
		ID3D11DeviceContext& context, 
		IDXGISwapChain1& swapChain, 
		const PIPELINE_DESC& shader_desc);
	void Shutdown();

	void SetShadowMapRender(ID3D11DeviceContext& deviceContext);
	void SetBackBufferRender(ID3D11DeviceContext& deviceContext);

	void UpdatePerConfig(ID3D11DeviceContext& deviceContext);
	void UpdatePerFrame(ID3D11DeviceContext& deviceContext);

	void UpdateVSPerFrame(PerFrameDataVS& data);
	void UpdateVSPerDrawCall(PerDrawCallDataVS& data);
	void UpdatePSPerFrame(PerFrameDataPS& data);
	void UpdatePSPerDrawCall(PerDrawCallDataPS& data);

	void BindConstantBuffers(ID3D11DeviceContext& deviceContext);
private:
	D3D11PipelineState(
		ID3D11Device& device,
		ID3D11DeviceContext& context,
		IDXGISwapChain1& swapChain, 
		const PIPELINE_DESC& shader_desc);

	float m_clearColor[4];

	ID3D11InputLayout* m_pIL;
	ID3D11VertexShader* m_pVS;
	ID3D11PixelShader* m_pPS;
	ID3D11RasterizerState* m_pRasterizerState;
	ID3D11DepthStencilState* m_pDepthStencilState;

	ID3D11RenderTargetView* m_pRenderTargetView;
	ID3D11DepthStencilView* m_pDepthStencilView;

	D3D11ConstantBuffer* m_pPerFrameCBufferVS;
	D3D11ConstantBuffer* m_pPerDrawCallCBufferVS;
	D3D11ConstantBuffer* m_pPerFrameCBufferPS;
	D3D11ConstantBuffer* m_pPerDrawCallCBufferPS;

	PerFrameDataVS m_perFrameDataVS;
	PerDrawCallDataVS m_perDrawCallDataVS;
	PerFrameDataPS m_perFrameDataPS;
	PerDrawCallDataPS m_perDrawCallDataPS;

	unsigned int m_cbufferVSRegCounter;
	unsigned int m_cbufferPSRegCounter;

	unsigned int m_textureVSRegCounter;
	unsigned int m_texturePSRegCounter;

	unsigned int m_samplerVSRegCounter;
	unsigned int m_samplerPSRegCounter;

	// Direct Illumination
	D3D11RenderTexture* m_pShadowMap;
	ID3D11SamplerState* m_pSampleStateWrap;

	// Indirect Illumination (Final Gathering)
	ID3D11Texture2D* m_pBufferA;
	ID3D11Texture2D* m_pBufferB;
	ID3D11Texture2D* m_pBufferC;
	ID3D11Texture2D* m_pBufferD;
	ID3D11Texture2D* m_pBufferE;
};

