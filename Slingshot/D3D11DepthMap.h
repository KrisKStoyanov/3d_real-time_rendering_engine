#pragma once
#include "D3D11Context.h"

struct PerFrameDataGS_DM
{
	DirectX::XMMATRIX viewMatrix[6];
	DirectX::XMMATRIX projectionMatrix;
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

class D3D11DepthMap : public D3D11PipelineState
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
		return m_pShadowMapSRV;
	}
private:
	D3D11DepthMap(D3D11Context& context);

	ID3D11InputLayout* m_pIL;
	ID3D11VertexShader* m_pVS;
	ID3D11GeometryShader* m_pGS;
	ID3D11PixelShader* m_pPS;

	D3D11_VIEWPORT m_viewport;

	ID3D11RasterizerState* m_pRasterizerState;

	ID3D11ShaderResourceView* m_pShadowMapSRV;
	ID3D11DepthStencilView* m_pShadowMapDSV;

	D3D11ConstantBuffer* m_pPerFrameCBufferGS;
	D3D11ConstantBuffer* m_pPerFrameCBufferVS;
	D3D11ConstantBuffer* m_pPerDrawCallCBufferVS;

	PerFrameDataGS_DM m_perFrameDataGS;
	PerFrameDataVS_DM m_perFrameDataVS;
	PerDrawCallDataVS_DM m_perDrawCallDataVS;

	unsigned int m_cbufferVSRegCounter;
	unsigned int m_cbufferGSRegCounter;
};

