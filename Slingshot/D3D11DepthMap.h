#pragma once
#include "D3D11Context.h"

class D3D11DepthMap //: public D3D11PipelineState
{
public:
	static D3D11DepthMap* Create(D3D11Context& context);
	void Shutdown();

	void UpdatePerConfig(ID3D11DeviceContext& deviceContext);
	void UpdatePerFrame(ID3D11DeviceContext& deviceContext);

	void UpdateBuffersPerFrame(PerFrameDataVS_DM& data);
	void UpdateBuffersPerDrawCall(PerDrawCallDataVS_DM& data);
	void BindConstantBuffers(ID3D11DeviceContext& deviceContext);

	inline ID3D11ShaderResourceView* GetShaderResourceView()
	{
		return m_pShaderResourceView;
	}
private:
	D3D11DepthMap(D3D11Context& context);

	float m_clearColor[4];

	ID3D11InputLayout* m_pIL;
	ID3D11VertexShader* m_pVS;
	ID3D11PixelShader* m_pPS;

	ID3D11RasterizerState* m_pRasterizerState;
	ID3D11DepthStencilState* m_pDepthStencilState;

	ID3D11RenderTargetView* m_pRenderTargetView;
	ID3D11DepthStencilView* m_pDepthStencilView;
	ID3D11ShaderResourceView* m_pShaderResourceView;

	D3D11ConstantBuffer* m_pPerFrameCBufferVS;
	D3D11ConstantBuffer* m_pPerDrawCallCBufferVS;

	PerFrameDataVS_DM m_perFrameDataVS;
	PerDrawCallDataVS_DM m_perDrawCallDataVS;

	unsigned int m_cbufferVSRegCounter;
};

