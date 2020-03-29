#pragma once
#include "PipelineState.h"
#include "D3D11Buffer.h"

class D3D11PipelineState : public PipelineState
{
public:
	static D3D11PipelineState* Create(ID3D11Device& device, const PIPELINE_DESC& shader_desc);
	void Shutdown();

	void UpdateVSPerFrame(
		DirectX::XMMATRIX viewMatrix,
		DirectX::XMMATRIX projMatrix);
	void UpdatePSPerFrame(
		DirectX::XMVECTOR cameraPos,
		DirectX::XMVECTOR lightPos,
		DirectX::XMFLOAT4 lightColor);
	void UpdateVSPerModel(DirectX::XMMATRIX worldMatrix);
	void UpdatePSPerModel(DirectX::XMFLOAT4 surfaceColor, float roughness);

	void Bind(ID3D11DeviceContext& deviceContext);
	void BindConstantBuffers(ID3D11DeviceContext& deviceContext);
private:
	D3D11PipelineState(ID3D11Device& device, const PIPELINE_DESC& shader_desc);

	ID3D11VertexShader* m_pVS;
	ID3D11PixelShader* m_pPS;
	ID3D11InputLayout* m_pIL;

	D3D11ConstantBuffer* m_pVS_WVP_CBuffer;
	D3D11ConstantBuffer* m_pPS_WorldTransform_CBuffer;
	D3D11ConstantBuffer* m_pPS_Light_CBuffer;
	D3D11ConstantBuffer* m_pPS_Material_CBuffer;

	WVPData m_wvpData;
	WorldTransformData m_worldTransformData;
	LightData m_lightData;
	MaterialData m_materialData;
};

