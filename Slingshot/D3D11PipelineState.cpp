#include "D3D11PipelineState.h"

D3D11PipelineState* D3D11PipelineState::Create(
	ID3D11Device& device,
	ID3D11DeviceContext& context,
	IDXGISwapChain1& swapChain, 
	const PIPELINE_DESC& pipeline_desc)
{
	return new D3D11PipelineState(device, context, swapChain, pipeline_desc);
}

D3D11PipelineState::D3D11PipelineState(
	ID3D11Device& device,
	ID3D11DeviceContext& context,
	IDXGISwapChain1& swapChain, 
	const PIPELINE_DESC& desc) :
	m_pVS(nullptr), m_pPS(nullptr), m_pIL(nullptr), m_cbufferVSRegCounter(0), m_cbufferPSRegCounter(0),
	m_wvpData(), m_perFrameData(), m_perDrawCallData()
{
	m_clearColor[0] = 0.0f;
	m_clearColor[1] = 0.0f;
	m_clearColor[2] = 0.0f;
	m_clearColor[3] = 1.0f;

	char* VS_bytecode = nullptr, * PS_bytecode = nullptr;
	size_t VS_size, PS_size;
	VS_bytecode = GetFileBytecode(desc.VS_filename, VS_size);
	PS_bytecode = GetFileBytecode(desc.PS_filename, PS_size);

	device.CreateVertexShader(VS_bytecode, VS_size, nullptr, &m_pVS);
	device.CreatePixelShader(PS_bytecode, PS_size, nullptr, &m_pPS);

	switch (desc.shadingModel)
	{
	case ShadingModel::GoochShading:
	{
		D3D11_INPUT_ELEMENT_DESC VS_inputLayout[2];

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

		device.CreateInputLayout(VS_inputLayout, 2, VS_bytecode, VS_size, &m_pIL);
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

		device.CreateInputLayout(VS_inputLayout, 2, VS_bytecode, VS_size, &m_pIL);
	}
	case ShadingModel::FinalGathering:
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

		device.CreateInputLayout(VS_inputLayout, 2, VS_bytecode, VS_size, &m_pIL);
	}
	break;
	}
	SAFE_DELETE_ARRAY(VS_bytecode);
	SAFE_DELETE_ARRAY(PS_bytecode);
	SetShadingModel(desc.shadingModel);

	CONSTANT_BUFFER_DESC desc0;
	desc0.cbufferData = &m_wvpData;
	desc0.cbufferSize = sizeof(m_wvpData);
	desc0.shaderType = ShaderType::VERTEX_SHADER;
	desc0.registerSlot = m_cbufferVSRegCounter;
	m_pVS_WVP_CBuffer = D3D11ConstantBuffer::Create(device, desc0);
	m_cbufferVSRegCounter++;

	CONSTANT_BUFFER_DESC desc1;
	desc1.cbufferData = &m_perFrameData;
	desc1.cbufferSize = sizeof(m_perFrameData); //sizeof(m_perFrameData) inaccurate interpretation - pending fix
	desc1.shaderType = ShaderType::PIXEL_SHADER;
	desc1.registerSlot = m_cbufferPSRegCounter;
	m_pPS_PerFrameCBuffer = D3D11ConstantBuffer::Create(device, desc1);
	m_cbufferPSRegCounter++;

	CONSTANT_BUFFER_DESC desc2;
	desc2.cbufferData = &m_perDrawCallData;
	desc2.cbufferSize = sizeof(m_perDrawCallData);
	desc2.shaderType = ShaderType::PIXEL_SHADER;
	desc2.registerSlot = m_cbufferPSRegCounter;
	m_pPS_PerDrawCallCBuffer = D3D11ConstantBuffer::Create(device, desc2);
	m_cbufferPSRegCounter++;

	DX::ThrowIfFailed(swapChain.GetBuffer(0, IID_PPV_ARGS(&m_pBackBuffer)));
	if (m_pBackBuffer != nullptr) {
		DX::ThrowIfFailed(device.CreateRenderTargetView(m_pBackBuffer, nullptr, &m_pRenderTargetView));
		//m_pBackBuffer->Release();
	}

	DXGI_SWAP_CHAIN_DESC1 swap_chain_desc;
	swapChain.GetDesc1(&swap_chain_desc);
	D3D11_TEXTURE2D_DESC depthStencilBufferDesc;
	ZeroMemory(&depthStencilBufferDesc, sizeof(depthStencilBufferDesc));
	depthStencilBufferDesc.Width = swap_chain_desc.Width;
	depthStencilBufferDesc.Height = swap_chain_desc.Height;
	depthStencilBufferDesc.MipLevels = 1;
	depthStencilBufferDesc.ArraySize = 1;
	depthStencilBufferDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depthStencilBufferDesc.SampleDesc.Count = 1;
	depthStencilBufferDesc.SampleDesc.Quality = 0;
	depthStencilBufferDesc.Usage = D3D11_USAGE_DEFAULT;
	depthStencilBufferDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	depthStencilBufferDesc.CPUAccessFlags = 0;
	depthStencilBufferDesc.MiscFlags = 0;
	DX::ThrowIfFailed(device.CreateTexture2D(&depthStencilBufferDesc, nullptr, &m_pDepthStencilBuffer));

	if (m_pDepthStencilBuffer)
	{
		D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
		ZeroMemory(&depthStencilViewDesc, sizeof(depthStencilViewDesc));
		depthStencilViewDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
		depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
		depthStencilViewDesc.Texture2D.MipSlice = 0;

		DX::ThrowIfFailed(
			device.CreateDepthStencilView(
				m_pDepthStencilBuffer,
				&depthStencilViewDesc,
				&m_pDepthStencilView));
	}

	D3D11_RASTERIZER_DESC rasterizer_desc;
	rasterizer_desc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
	rasterizer_desc.CullMode = D3D11_CULL_MODE::D3D11_CULL_NONE; //overhaul sphere engine creation method and switch to backface culling
	rasterizer_desc.FrontCounterClockwise = false;
	rasterizer_desc.DepthBias = false;
	rasterizer_desc.DepthBiasClamp = 0;
	rasterizer_desc.SlopeScaledDepthBias = 0;
	rasterizer_desc.DepthClipEnable = true;
	rasterizer_desc.ScissorEnable = false;
	rasterizer_desc.MultisampleEnable = false;
	rasterizer_desc.AntialiasedLineEnable = false;
	//rasterizer_desc.ForcedSampleCount = 0; <-featured in D3D11_RASTERIZER_DESC1 (requires device1), future update consideration
	device.CreateRasterizerState(&rasterizer_desc, &m_pRasterizerState);
}

void D3D11PipelineState::Shutdown()
{
	SAFE_RELEASE(m_pVS);
	SAFE_RELEASE(m_pPS);
	SAFE_RELEASE(m_pIL);
	SAFE_RELEASE(m_pRasterizerState);

	SAFE_DESTROY(m_pVS_WVP_CBuffer);
	SAFE_DESTROY(m_pPS_PerFrameCBuffer);
	SAFE_DESTROY(m_pPS_PerDrawCallCBuffer);
}

void D3D11PipelineState::StartFrameRender(ID3D11DeviceContext& deviceContext)
{
	deviceContext.ClearRenderTargetView(m_pRenderTargetView, m_clearColor);
	deviceContext.ClearDepthStencilView(m_pDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 1);
	deviceContext.OMSetRenderTargets(1,
		&m_pRenderTargetView,
		m_pDepthStencilView);
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
	m_perFrameData.camPos = cameraPos;
	m_perFrameData.lightPos = lightPos;
	m_perFrameData.lightColor = lightColor;
}

void D3D11PipelineState::UpdatePerModel(
	DirectX::XMMATRIX worldMatrix,
	DirectX::XMFLOAT4 surfaceColor, 
	float roughness)
{
	m_wvpData.worldMatrix = worldMatrix;
	m_perDrawCallData.surfaceColor = surfaceColor;
	m_perDrawCallData.roughness = roughness;
}

void D3D11PipelineState::Bind(ID3D11DeviceContext& deviceContext)
{
	deviceContext.IASetInputLayout(m_pIL);
	deviceContext.VSSetShader(m_pVS, nullptr, 0);
	deviceContext.PSSetShader(m_pPS, nullptr, 0);
	deviceContext.RSSetState(m_pRasterizerState);
}

void D3D11PipelineState::BindConstantBuffers(ID3D11DeviceContext& deviceContext)
{
	m_pVS_WVP_CBuffer->Bind(deviceContext, &m_wvpData);
	m_pPS_PerFrameCBuffer->Bind(deviceContext, &m_perFrameData);
	m_pPS_PerDrawCallCBuffer->Bind(deviceContext, &m_perDrawCallData);
}