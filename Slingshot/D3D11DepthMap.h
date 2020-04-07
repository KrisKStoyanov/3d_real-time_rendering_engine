#pragma once
#include "D3D11Context.h"

struct PerFrameDataGS_DM
{
	DirectX::XMMATRIX viewMatrix0;
	DirectX::XMMATRIX viewMatrix1;
	DirectX::XMMATRIX viewMatrix2;
	DirectX::XMMATRIX viewMatrix3;
	DirectX::XMMATRIX viewMatrix4;
	DirectX::XMMATRIX viewMatrix5;

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

	void UpdateBuffersPerFrame(PerFrameDataGS_DM& data);
	void UpdateBuffersPerDrawCall(PerDrawCallDataVS_DM& data);
	void BindConstantBuffers(ID3D11DeviceContext& deviceContext);

	inline ID3D11ShaderResourceView* GetShaderResourceView()
	{
		return m_pDepthMapsSRV[0];
	}
private:
	D3D11DepthMap(D3D11Context& context);

	float m_clearColor[4];

	ID3D11InputLayout* m_pIL;
	ID3D11VertexShader* m_pVS;
	ID3D11GeometryShader* m_pGS;
	ID3D11PixelShader* m_pPS;

	ID3D11RasterizerState* m_pRasterizerState;
	ID3D11DepthStencilState* m_pDepthStencilState;

	ID3D11RenderTargetView* m_pDepthMapsRTV[6];
	ID3D11ShaderResourceView* m_pDepthMapsSRV[6];

	ID3D11DepthStencilView* m_pDepthStencilView;

	D3D11ConstantBuffer* m_pPerFrameCBufferGS;
	D3D11ConstantBuffer* m_pPerDrawCallCBufferVS;

	PerFrameDataGS_DM m_perFrameDataGS;
	PerDrawCallDataVS_DM m_perDrawCallDataVS;

	unsigned int m_cbufferVSRegCounter;
	unsigned int m_cbufferGSRegCounter;
};

