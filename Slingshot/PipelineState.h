#pragma once
#include "D3D11Context.h" //included twice (again in Mesh.h)
#include "Vertex.h"
#include "FileParsing.h"

struct PIPELINE_DESC
{
	const char* VS_filename;
	const char* PS_filename;
};

class PipelineState {
public:
	static PipelineState* Create(D3D11Context& graphicsContext, PIPELINE_DESC& shader_desc, ShadingModel shadingModel);
	void Shutdown();

	ID3D11VertexShader* GetVertexShader();
	ID3D11PixelShader* GetPixelShader();
	ID3D11InputLayout* GetInputLayout();

	ShadingModel GetShadingModel();
private:
	PipelineState(D3D11Context& graphicsContext, PIPELINE_DESC& shader_desc, ShadingModel shadingModel);

	ID3D11VertexShader* m_pVS;
	ID3D11PixelShader* m_pPS;
	ID3D11InputLayout* m_pIL;

	ShadingModel m_shadingModel;
};