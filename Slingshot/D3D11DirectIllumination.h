#pragma once
#include "D3D11Context.h"

// Holds BackBuffer RTV - might be more optimal to store in D3D11Context for accessibility
class D3D11DirectIllumination //: public D3D11PipelineState
{
public:
	static D3D11DirectIllumination* Create(D3D11Context& context);
	void Shutdown();

	void UpdatePerConfig(ID3D11DeviceContext& deviceContext);
	void UpdatePerFrame(ID3D11DeviceContext& deviceContext, ID3D11ShaderResourceView* depthMap);

	void UpdateBuffersPerFrame(PerFrameDataVS_DI& dataVS, PerFrameDataPS_DI& dataPS);
	void UpdateBuffersPerDrawCall(PerDrawCallDataVS_DI& dataVS, PerDrawCallDataPS_DI& dataPS);
	void BindConstantBuffers(ID3D11DeviceContext& deviceContext);

	void EndFrameRender(ID3D11DeviceContext& deviceContext);

private:
	D3D11DirectIllumination(D3D11Context& context);

	float m_clearColor[4];

	ID3D11InputLayout* m_pIL;
	ID3D11VertexShader* m_pVS;
	ID3D11PixelShader* m_pPS;

	ID3D11RasterizerState* m_pRasterizerState;
	ID3D11DepthStencilState* m_pDepthStencilState;

	ID3D11RenderTargetView* m_pRenderTargetView;
	ID3D11DepthStencilView* m_pDepthStencilView;

	ID3D11SamplerState* m_pSampleStateWrap;

	D3D11ConstantBuffer* m_pPerFrameCBufferVS;
	D3D11ConstantBuffer* m_pPerDrawCallCBufferVS;
	D3D11ConstantBuffer* m_pPerFrameCBufferPS;
	D3D11ConstantBuffer* m_pPerDrawCallCBufferPS;

	PerFrameDataVS_DI m_perFrameDataVS;
	PerDrawCallDataVS_DI m_perDrawCallDataVS;
	PerFrameDataPS_DI m_perFrameDataPS;
	PerDrawCallDataPS_DI m_perDrawCallDataPS;

	unsigned int m_cbufferVSRegCounter;
	unsigned int m_cbufferPSRegCounter;

	unsigned int m_samplerPSRegCounter;
};

