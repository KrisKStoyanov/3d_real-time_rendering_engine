#pragma once
#include "PipelineState.h"
#include "D3D11Buffer.h"

struct WVPData //vertex 
{
	DirectX::XMMATRIX worldMatrix;
	DirectX::XMMATRIX viewMatrix;
	DirectX::XMMATRIX projMatrix;
};

struct PerFrameData //pixel 
{
	DirectX::XMVECTOR camPos;
	DirectX::XMVECTOR lightPos;
	DirectX::XMFLOAT4 lightColor;
};

struct PerDrawCallData //pixel 
{
	DirectX::XMFLOAT4 surfaceColor;
	float roughness;
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

	void StartFrameRender(ID3D11DeviceContext& deviceContext);

	void UpdatePerFrame(
		DirectX::XMMATRIX viewMatrix,
		DirectX::XMMATRIX projMatrix,
		DirectX::XMVECTOR cameraPos,
		DirectX::XMVECTOR lightPos,
		DirectX::XMFLOAT4 lightColor);
	void UpdatePerModel(
		DirectX::XMMATRIX worldMatrix,
		DirectX::XMFLOAT4 surfaceColor, 
		float roughness);

	void Bind(ID3D11DeviceContext& deviceContext);
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

	ID3D11Texture2D* m_pBackBuffer;
	ID3D11RenderTargetView* m_pRenderTargetView;
	ID3D11Texture2D* m_pDepthStencilBuffer;
	ID3D11DepthStencilView* m_pDepthStencilView;

	D3D11ConstantBuffer* m_pVS_WVP_CBuffer;
	D3D11ConstantBuffer* m_pPS_PerFrameCBuffer;
	D3D11ConstantBuffer* m_pPS_PerDrawCallCBuffer;

	WVPData m_wvpData;
	PerFrameData m_perFrameData;
	PerDrawCallData m_perDrawCallData;

	unsigned int m_cbufferVSRegCounter;
	unsigned int m_cbufferPSRegCounter;
};

