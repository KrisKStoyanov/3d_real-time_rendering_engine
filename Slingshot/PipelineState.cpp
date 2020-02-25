#include "PipelineState.h"

PipelineState* PipelineState::Create(D3D11Context& graphicsContext, PIPELINE_DESC& shader_desc, ShadingModel shadingModel)
{
	return new PipelineState(graphicsContext, shader_desc, shadingModel);
}

PipelineState::PipelineState(D3D11Context& graphicsContext, PIPELINE_DESC& pipeline_desc, ShadingModel shadingModel) :
	m_pVS(nullptr), m_pPS(nullptr), m_pIL(nullptr), m_shadingModel(shadingModel)
{
	char* ColorVS_bytecode = nullptr, * ColorPS_bytecode = nullptr;
	size_t ColorVS_size, ColorPS_size;
	ColorVS_bytecode = GetFileBytecode(pipeline_desc.VS_filename, ColorVS_size);
	ColorPS_bytecode = GetFileBytecode(pipeline_desc.PS_filename, ColorPS_size);

	graphicsContext.GetDevice()->CreateVertexShader(ColorVS_bytecode, ColorVS_size, nullptr, &m_pVS);
	graphicsContext.GetDevice()->CreatePixelShader(ColorPS_bytecode, ColorPS_size, nullptr, &m_pPS);

	switch (shadingModel)
	{
	case ShadingModel::GoochShading:
	{
		D3D11_INPUT_ELEMENT_DESC VS_inputLayout[3];

		VS_inputLayout[0].SemanticName = "POSITION";
		VS_inputLayout[0].SemanticIndex = 0;
		VS_inputLayout[0].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
		VS_inputLayout[0].InputSlot = 0;
		VS_inputLayout[0].AlignedByteOffset = 0;
		VS_inputLayout[0].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
		VS_inputLayout[0].InstanceDataStepRate = 0;

		VS_inputLayout[1].SemanticName = "COLOR";
		VS_inputLayout[1].SemanticIndex = 0;
		VS_inputLayout[1].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
		VS_inputLayout[1].InputSlot = 0;
		VS_inputLayout[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
		VS_inputLayout[1].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
		VS_inputLayout[1].InstanceDataStepRate = 0;

		VS_inputLayout[2].SemanticName = "NORMAL";
		VS_inputLayout[2].SemanticIndex = 0;
		VS_inputLayout[2].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
		VS_inputLayout[2].InputSlot = 0;
		VS_inputLayout[2].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
		VS_inputLayout[2].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
		VS_inputLayout[2].InstanceDataStepRate = 0;

		graphicsContext.GetDevice()->CreateInputLayout(VS_inputLayout, 3, ColorVS_bytecode, ColorVS_size, &m_pIL);
	}
	break;
	}

	SAFE_DELETE_ARRAY(ColorVS_bytecode);
	SAFE_DELETE_ARRAY(ColorPS_bytecode);
}

void PipelineState::Shutdown()
{
	SAFE_RELEASE(m_pVS);
	SAFE_RELEASE(m_pPS);
	SAFE_RELEASE(m_pIL);
}

ID3D11VertexShader* PipelineState::GetVertexShader()
{
	return m_pVS;
}

ID3D11PixelShader* PipelineState::GetPixelShader()
{
	return m_pPS;
}

ID3D11InputLayout* PipelineState::GetInputLayout()
{
	return m_pIL;
}

ShadingModel PipelineState::GetShadingModel()
{
	return m_shadingModel;
}
