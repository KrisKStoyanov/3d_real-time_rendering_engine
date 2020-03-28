#pragma once
#include "Mesh.h"
#include "FileParsing.h"

struct PIPELINE_DESC
{
	ShadingModel shadingModel;
	const char* VS_filename;
	const char* PS_filename;
};

class PipelineState {
public:
	static PipelineState* Create(D3D11Context& graphicsContext, const PIPELINE_DESC& shader_desc);
	void Shutdown();

	void UpdateVSPerFrame(
		DirectX::XMMATRIX viewMatrix, 
		DirectX::XMMATRIX projMatrix);
	void UpdatePSPerFrame(
		DirectX::XMVECTOR cameraPos, 
		DirectX::XMVECTOR lightPos, 
		DirectX::XMFLOAT4 lightColor);
	void UpdateVSPerEntity(DirectX::XMMATRIX worldMatrix);
	void UpdatePSPerEntity(DirectX::XMFLOAT4 surfaceColor, float roughness);

	void Bind(ID3D11DeviceContext& deviceContext);
	void BindConstantBuffers(ID3D11DeviceContext& deviceContext);

	ShadingModel GetShadingModel();
private:
	PipelineState(D3D11Context& graphicsContext, const PIPELINE_DESC& shader_desc);

	ID3D11VertexShader* m_pVS;
	ID3D11PixelShader* m_pPS;
	ID3D11InputLayout* m_pIL;

	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pVS_WVP_CBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pPS_WorldTransform_CBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pPS_Light_CBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pPS_Material_CBuffer;

	gfx::WVPData m_wvpData;
	gfx::WorldTransformData m_worldTransformData;
	gfx::LightData m_lightData;
	gfx::MaterialData m_materialData;

	ShadingModel m_shadingModel;
};