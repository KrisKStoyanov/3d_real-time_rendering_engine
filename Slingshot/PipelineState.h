#pragma once
#include "D3D11Context.h" //included twice (again in Mesh.h)
#include "Material.h"
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

	const Microsoft::WRL::ComPtr<ID3D11Buffer> GetVSCB();
	const Microsoft::WRL::ComPtr<ID3D11Buffer> GetPSCB();

	ID3D11VertexShader* GetVertexShader();
	ID3D11PixelShader* GetPixelShader();
	ID3D11InputLayout* GetInputLayout();

	ShadingModel GetShadingModel();
private:
	PipelineState(D3D11Context& graphicsContext, const PIPELINE_DESC& shader_desc);

	ID3D11VertexShader* m_pVS;
	ID3D11PixelShader* m_pPS;
	ID3D11InputLayout* m_pIL;

	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pVSCB;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pPSCB;

	ShadingModel m_shadingModel;
};