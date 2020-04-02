#include "D3D11PipelineState.h"

D3D11PipelineState* D3D11PipelineState::Create(ID3D11Device& device, const PIPELINE_DESC& pipeline_desc)
{
	return new D3D11PipelineState(device, pipeline_desc);
}

D3D11PipelineState::D3D11PipelineState(ID3D11Device& device, const PIPELINE_DESC& pipeline_desc) :
	m_pVS(nullptr), m_pPS(nullptr), m_pIL(nullptr), m_cbufferVSRegCounter(0), m_cbufferPSRegCounter(0)
{
	char* ColorVS_bytecode = nullptr, * ColorPS_bytecode = nullptr;
	size_t ColorVS_size, ColorPS_size;
	ColorVS_bytecode = GetFileBytecode(pipeline_desc.VS_filename, ColorVS_size);
	ColorPS_bytecode = GetFileBytecode(pipeline_desc.PS_filename, ColorPS_size);

	device.CreateVertexShader(ColorVS_bytecode, ColorVS_size, nullptr, &m_pVS);
	device.CreatePixelShader(ColorPS_bytecode, ColorPS_size, nullptr, &m_pPS);

	switch (pipeline_desc.shadingModel)
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

		device.CreateInputLayout(VS_inputLayout, 2, ColorVS_bytecode, ColorVS_size, &m_pIL);
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

		device.CreateInputLayout(VS_inputLayout, 2, ColorVS_bytecode, ColorVS_size, &m_pIL);
	}
	break;
	}

	SetShadingModel(pipeline_desc.shadingModel);

	CONSTANT_BUFFER_DESC desc0;
	desc0.cbufferData = &m_wvpData;
	desc0.cbufferSize = sizeof(m_wvpData);
	desc0.shaderType = ShaderType::VERTEX_SHADER;
	desc0.registerSlot = m_cbufferVSRegCounter;
	m_pVS_WVP_CBuffer = D3D11ConstantBuffer::Create(device, desc0);
	m_cbufferVSRegCounter++;

	CONSTANT_BUFFER_DESC desc1;
	desc1.cbufferData = &m_worldTransformData;
	desc1.cbufferSize = 64; //sizeof(m_worldTransformData) inaccurate interpretation - pending fix
	desc1.shaderType = ShaderType::PIXEL_SHADER;
	desc1.registerSlot = m_cbufferPSRegCounter;
	m_pPS_WorldTransform_CBuffer = D3D11ConstantBuffer::Create(device, desc1);
	m_cbufferPSRegCounter++;

	CONSTANT_BUFFER_DESC desc2;
	desc2.cbufferData = &m_lightData;
	desc2.cbufferSize = sizeof(m_lightData); 
	desc2.shaderType = ShaderType::PIXEL_SHADER;
	desc2.registerSlot = m_cbufferPSRegCounter;
	m_pPS_Light_CBuffer = D3D11ConstantBuffer::Create(device, desc2);
	m_cbufferPSRegCounter++;

	CONSTANT_BUFFER_DESC desc3;
	desc3.cbufferData = &m_materialData; 
	desc3.cbufferSize = sizeof(m_materialData); 
	desc3.shaderType = ShaderType::PIXEL_SHADER;
	desc3.registerSlot = m_cbufferPSRegCounter;
	m_pPS_Material_CBuffer = D3D11ConstantBuffer::Create(device, desc3);
	m_cbufferPSRegCounter++;

	SAFE_DELETE_ARRAY(ColorVS_bytecode);
	SAFE_DELETE_ARRAY(ColorPS_bytecode);
}

void D3D11PipelineState::Shutdown()
{
	SAFE_RELEASE(m_pVS);
	SAFE_RELEASE(m_pPS);
	SAFE_RELEASE(m_pIL);

	SAFE_DESTROY(m_pVS_WVP_CBuffer);
	SAFE_DESTROY(m_pPS_WorldTransform_CBuffer);
	SAFE_DESTROY(m_pPS_Light_CBuffer);
	SAFE_DESTROY(m_pPS_Material_CBuffer);
}

void D3D11PipelineState::UpdatePerFrame(
	DirectX::XMMATRIX viewMatrix, 
	DirectX::XMMATRIX projMatrix,
	DirectX::XMVECTOR cameraPos, 
	DirectX::XMVECTOR lightPos, 
	DirectX::XMFLOAT4 lightColor)
{
	m_wvpData.viewMatrix = viewMatrix;
	m_wvpData.projMatrix = projMatrix;
	m_worldTransformData.camPos = cameraPos;
	m_lightData.lightPos = lightPos;
	m_lightData.lightColor = lightColor;
}

void D3D11PipelineState::UpdatePerModel(
	DirectX::XMMATRIX worldMatrix,
	DirectX::XMFLOAT4 surfaceColor, 
	float roughness)
{
	m_wvpData.worldMatrix = worldMatrix;
	m_materialData.surfaceColor = surfaceColor;
	m_materialData.roughness = roughness;
}

void D3D11PipelineState::Bind(ID3D11DeviceContext& deviceContext)
{
	deviceContext.IASetInputLayout(m_pIL);
	deviceContext.VSSetShader(m_pVS, nullptr, 0);
	deviceContext.PSSetShader(m_pPS, nullptr, 0);
}

void D3D11PipelineState::BindConstantBuffers(ID3D11DeviceContext& deviceContext)
{
	m_pVS_WVP_CBuffer->Bind(deviceContext, &m_wvpData);
	m_pPS_WorldTransform_CBuffer->Bind(deviceContext, &m_worldTransformData);
	m_pPS_Light_CBuffer->Bind(deviceContext, &m_lightData);
	m_pPS_Material_CBuffer->Bind(deviceContext, &m_materialData);
}