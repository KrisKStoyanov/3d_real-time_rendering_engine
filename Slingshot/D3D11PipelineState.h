#pragma once
#include "PipelineState.h"
#include "D3D11RenderTexture.h"

struct PerFrameDataVS_DI
{
	DirectX::XMMATRIX cameraViewMatrix;
	DirectX::XMMATRIX cameraProjMatrix;
	DirectX::XMMATRIX lightViewMatrix;
	DirectX::XMMATRIX lightProjMatrix;
	DirectX::XMVECTOR lightPos;
};

struct PerDrawCallDataVS_DI
{
	DirectX::XMMATRIX worldMatrix;
};

struct PerFrameDataPS_DI
{
	DirectX::XMFLOAT4 ambientColor; 
	DirectX::XMFLOAT4 diffuseColor; 
};

struct PerDrawCallDataPS_DI 
{
	DirectX::XMFLOAT4 surfaceColor;
};

struct PerFrameDataVS_DM
{
	DirectX::XMMATRIX viewMatrix;
	DirectX::XMMATRIX projectionMatrix;
};

struct PerDrawCallDataVS_DM
{
	DirectX::XMMATRIX worldMatrix;
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

	void SetDepthMapRender(ID3D11DeviceContext& deviceContext);

	void UpdatePerConfig(ID3D11DeviceContext& deviceContext);
	void UpdatePerFrame_DM(ID3D11DeviceContext& deviceContext);
	void UpdatePerFrame_DI(ID3D11DeviceContext& deviceContext);

	void BindShaderResources(ID3D11DeviceContext& deviceContext);
	void UnbindShaderResources(ID3D11DeviceContext& deviceContext);

	void UpdateVSPerFrame_DM(PerFrameDataVS_DM& data);
	void UpdateVSPerDrawCall_DM(PerDrawCallDataVS_DM& data);

	void UpdateVSPerFrame_DI(PerFrameDataVS_DI& data);
	void UpdateVSPerDrawCall_DI(PerDrawCallDataVS_DI& data);
	void UpdatePSPerFrame_DI(PerFrameDataPS_DI& data);
	void UpdatePSPerDrawCall_DI(PerDrawCallDataPS_DI& data);

	void BindConstantBuffers_DI(ID3D11DeviceContext& deviceContext);
	void BindConstantBuffers_DM(ID3D11DeviceContext& deviceContext);
private:
	D3D11PipelineState(
		ID3D11Device& device,
		ID3D11DeviceContext& context,
		IDXGISwapChain1& swapChain, 
		const PIPELINE_DESC& shader_desc);

	float m_clearColor[4];

	ID3D11RasterizerState* m_pRasterizerState;
	ID3D11DepthStencilState* m_pDepthStencilState;
	ID3D11DepthStencilView* m_pDepthStencilView;

	// Depth
	ID3D11InputLayout* m_pIL_DepthMap;
	ID3D11VertexShader* m_pVS_DepthMap;
	ID3D11PixelShader* m_pPS_DepthMap;

	D3D11RenderTexture* m_pDepthMap;
	D3D11ConstantBuffer* m_pPerFrameCBufferVS_DM;
	D3D11ConstantBuffer* m_pPerDrawCallCBufferVS_DM;

	PerFrameDataVS_DM m_perFrameDataVS_DM;
	PerDrawCallDataVS_DM m_perDrawCallDataVS_DM;

	unsigned int m_cbufferVSRegCounter_DM;

	// Direct Illumination
	ID3D11InputLayout* m_pIL_DirectIllumination;
	ID3D11VertexShader* m_pVS_DirectIllumination;
	ID3D11PixelShader* m_pPS_DirectIllumination;

	D3D11ConstantBuffer* m_pPerFrameCBufferVS_DI;
	D3D11ConstantBuffer* m_pPerDrawCallCBufferVS_DI;
	D3D11ConstantBuffer* m_pPerFrameCBufferPS_DI;
	D3D11ConstantBuffer* m_pPerDrawCallCBufferPS_DI;

	PerFrameDataVS_DI m_perFrameDataVS_DI;
	PerDrawCallDataVS_DI m_perDrawCallDataVS_DI;
	PerFrameDataPS_DI m_perFrameDataPS_DI;
	PerDrawCallDataPS_DI m_perDrawCallDataPS_DI;
	ID3D11SamplerState* m_pSampleStateWrap;

	unsigned int m_cbufferVSRegCounter_DI;
	unsigned int m_cbufferPSRegCounter_DI;

	unsigned int m_textureVSRegCounter_DI;
	unsigned int m_texturePSRegCounter_DI;

	unsigned int m_samplerVSRegCounter_DI;
	unsigned int m_samplerPSRegCounter_DI;

	// Indirect Illumination (Final Gathering)
	//ID3D11Texture2D* m_pBufferA;
	//ID3D11Texture2D* m_pBufferB;
	//ID3D11Texture2D* m_pBufferC;
	//ID3D11Texture2D* m_pBufferD;
	//ID3D11Texture2D* m_pBufferE;
};

