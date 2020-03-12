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

		VS_inputLayout[1].SemanticName = "NORMAL";
		VS_inputLayout[1].SemanticIndex = 0;
		VS_inputLayout[1].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
		VS_inputLayout[1].InputSlot = 0;
		VS_inputLayout[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
		VS_inputLayout[1].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
		VS_inputLayout[1].InstanceDataStepRate = 0;

		graphicsContext.GetDevice()->CreateInputLayout(VS_inputLayout, 2, ColorVS_bytecode, ColorVS_size, &m_pIL);
	}
	break;
	case ShadingModel::OrenNayarShading:
	{
		D3D11_INPUT_ELEMENT_DESC VS_inputLayout[3];

		VS_inputLayout[0].SemanticName = "POSITION";
		VS_inputLayout[0].SemanticIndex = 0;
		VS_inputLayout[0].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
		VS_inputLayout[0].InputSlot = 0;
		VS_inputLayout[0].AlignedByteOffset = 0;
		VS_inputLayout[0].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
		VS_inputLayout[0].InstanceDataStepRate = 0;

		VS_inputLayout[1].SemanticName = "NORMAL";
		VS_inputLayout[1].SemanticIndex = 0;
		VS_inputLayout[1].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
		VS_inputLayout[1].InputSlot = 0;
		VS_inputLayout[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
		VS_inputLayout[1].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
		VS_inputLayout[1].InstanceDataStepRate = 0;

		graphicsContext.GetDevice()->CreateInputLayout(VS_inputLayout, 2, ColorVS_bytecode, ColorVS_size, &m_pIL);
	}
	break;
	}

	VS_CONSTANT_BUFFER vs_cb;

	D3D11_BUFFER_DESC vs_cb_desc;
	ZeroMemory(&vs_cb_desc, sizeof(vs_cb_desc));
	vs_cb_desc.Usage = D3D11_USAGE_DEFAULT;
	vs_cb_desc.ByteWidth = 256; // sizeof(VS_CONSTANT_BUFFER) = 128 <- constant buffer size must be a multiple of 16 bytes;
	vs_cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	vs_cb_desc.CPUAccessFlags = 0;
	vs_cb_desc.MiscFlags = 0;
	vs_cb_desc.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vs_cb_data;
	vs_cb_data.pSysMem = &vs_cb;
	vs_cb_data.SysMemPitch = 0;
	vs_cb_data.SysMemSlicePitch = 0;

	DX::ThrowIfFailed(graphicsContext.GetDevice()->CreateBuffer(
		&vs_cb_desc, &vs_cb_data, m_pVSCB.GetAddressOf()));

	PS_CONSTANT_BUFFER ps_cb;

	D3D11_BUFFER_DESC ps_cb_desc;
	ZeroMemory(&ps_cb_desc, sizeof(ps_cb_desc));
	ps_cb_desc.Usage = D3D11_USAGE_DEFAULT;
	ps_cb_desc.ByteWidth = 80; // sizeof(VS_CONSTANT_BUFFER) = 128 <- constant buffer size must be a multiple of 16 bytes;
	ps_cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	ps_cb_desc.CPUAccessFlags = 0;
	ps_cb_desc.MiscFlags = 0;
	ps_cb_desc.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA ps_cb_data;
	ps_cb_data.pSysMem = &ps_cb;
	ps_cb_data.SysMemPitch = 0;
	ps_cb_data.SysMemSlicePitch = 0;

	DX::ThrowIfFailed(graphicsContext.GetDevice()->CreateBuffer(
		&ps_cb_desc, &ps_cb_data, m_pPSCB.GetAddressOf()));

	SAFE_DELETE_ARRAY(ColorVS_bytecode);
	SAFE_DELETE_ARRAY(ColorPS_bytecode);
}

void PipelineState::Shutdown()
{
	SAFE_RELEASE(m_pVS);
	SAFE_RELEASE(m_pPS);
	SAFE_RELEASE(m_pIL);
}

const Microsoft::WRL::ComPtr<ID3D11Buffer> PipelineState::GetVSCB()
{
	return m_pVSCB.Get();
}

const Microsoft::WRL::ComPtr<ID3D11Buffer> PipelineState::GetPSCB()
{
	return m_pPSCB.Get();
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
