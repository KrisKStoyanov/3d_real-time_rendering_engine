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
	m_pVS(nullptr), m_pPS(nullptr), m_pIL(nullptr), 
	m_cbufferVSRegCounter(0), m_cbufferPSRegCounter(0),
	m_samplerVSRegCounter(0), m_samplerPSRegCounter(0),
	m_textureVSRegCounter(0), m_texturePSRegCounter(0),
	m_perFrameDataVS(), m_perDrawCallDataVS(), m_perFrameDataPS(), m_perDrawCallDataPS(),
	m_pPerFrameCBufferVS(nullptr), m_pPerDrawCallCBufferVS(nullptr),
	m_pPerFrameCBufferPS(nullptr), m_pPerDrawCallCBufferPS(nullptr),
	m_pSampleStateClamp(nullptr), m_pSampleStateWrap(nullptr)
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

		VS_inputLayout[2].SemanticName = "TEXCOORD";
		VS_inputLayout[2].SemanticIndex = 0;
		VS_inputLayout[2].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32_FLOAT;
		VS_inputLayout[2].InputSlot = 0;
		VS_inputLayout[2].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
		VS_inputLayout[2].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
		VS_inputLayout[2].InstanceDataStepRate = 0;

		device.CreateInputLayout(VS_inputLayout, 3, VS_bytecode, VS_size, &m_pIL);
	}
	break;
	}
	SAFE_DELETE_ARRAY(VS_bytecode);
	SAFE_DELETE_ARRAY(PS_bytecode);
	SetShadingModel(desc.shadingModel);

	ID3D11Texture2D* pBackBuffer;
	DX::ThrowIfFailed(swapChain.GetBuffer(0, IID_PPV_ARGS(&pBackBuffer)));
	if (pBackBuffer != nullptr) {
		DX::ThrowIfFailed(device.CreateRenderTargetView(pBackBuffer, nullptr, &m_pRenderTargetView));
		SAFE_RELEASE(pBackBuffer);
	}

	DXGI_SWAP_CHAIN_DESC1 swapChainDesc;
	swapChain.GetDesc1(&swapChainDesc);

	ID3D11Texture2D* depthStencilBuffer;
	D3D11_TEXTURE2D_DESC depthStencilBufferDesc;
	ZeroMemory(&depthStencilBufferDesc, sizeof(depthStencilBufferDesc));
	depthStencilBufferDesc.Width = swapChainDesc.Width;
	depthStencilBufferDesc.Height = swapChainDesc.Height;
	depthStencilBufferDesc.MipLevels = 1;
	depthStencilBufferDesc.ArraySize = 1;
	depthStencilBufferDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depthStencilBufferDesc.SampleDesc.Count = 1;
	depthStencilBufferDesc.SampleDesc.Quality = 0;
	depthStencilBufferDesc.Usage = D3D11_USAGE_DEFAULT;
	depthStencilBufferDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	depthStencilBufferDesc.CPUAccessFlags = 0;
	depthStencilBufferDesc.MiscFlags = 0;
	DX::ThrowIfFailed(device.CreateTexture2D(&depthStencilBufferDesc, nullptr, &depthStencilBuffer));

	D3D11_DEPTH_STENCIL_DESC depthStencilStateDesc;
	depthStencilStateDesc.DepthEnable = true;
	depthStencilStateDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	depthStencilStateDesc.DepthFunc = D3D11_COMPARISON_LESS;
	depthStencilStateDesc.StencilEnable = true;
	depthStencilStateDesc.StencilReadMask = 0xFF;
	depthStencilStateDesc.StencilWriteMask = 0xFF;
	depthStencilStateDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	depthStencilStateDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_INCR;
	depthStencilStateDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	depthStencilStateDesc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	depthStencilStateDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	depthStencilStateDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_DECR;
	depthStencilStateDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	depthStencilStateDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	DX::ThrowIfFailed(device.CreateDepthStencilState(&depthStencilStateDesc, &m_pDepthStencilState));

	if (depthStencilBuffer)
	{
		D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
		ZeroMemory(&depthStencilViewDesc, sizeof(depthStencilViewDesc));
		depthStencilViewDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
		depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
		depthStencilViewDesc.Texture2D.MipSlice = 0;

		DX::ThrowIfFailed(
			device.CreateDepthStencilView(
				depthStencilBuffer,
				&depthStencilViewDesc,
				&m_pDepthStencilView));

		SAFE_RELEASE(depthStencilBuffer);
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

	CONSTANT_BUFFER_DESC desc0;
	desc0.cbufferData = &m_perFrameDataVS;
	desc0.cbufferSize = sizeof(m_perFrameDataVS);
	desc0.shaderType = ShaderType::VERTEX_SHADER;
	desc0.registerSlot = m_cbufferVSRegCounter;
	m_pPerFrameCBufferVS = D3D11ConstantBuffer::Create(device, desc0);
	m_cbufferVSRegCounter++;

	CONSTANT_BUFFER_DESC desc1;
	desc1.cbufferData = &m_perDrawCallDataVS;
	desc1.cbufferSize = sizeof(m_perDrawCallDataVS);
	desc1.shaderType = ShaderType::VERTEX_SHADER;
	desc1.registerSlot = m_cbufferVSRegCounter;
	m_pPerDrawCallCBufferVS = D3D11ConstantBuffer::Create(device, desc1);
	m_cbufferVSRegCounter++;

	CONSTANT_BUFFER_DESC desc2;
	desc2.cbufferData = &m_perFrameDataPS;
	desc2.cbufferSize = sizeof(m_perFrameDataPS);
	desc2.shaderType = ShaderType::PIXEL_SHADER;
	desc2.registerSlot = m_cbufferPSRegCounter;
	m_pPerFrameCBufferPS = D3D11ConstantBuffer::Create(device, desc2);
	m_cbufferPSRegCounter++;

	CONSTANT_BUFFER_DESC desc3;
	desc3.cbufferData = &m_perDrawCallDataPS;
	desc3.cbufferSize = sizeof(m_perDrawCallDataPS);
	desc3.shaderType = ShaderType::PIXEL_SHADER;
	desc3.registerSlot = m_cbufferPSRegCounter;
	m_pPerDrawCallCBufferPS = D3D11ConstantBuffer::Create(device, desc3);
	m_cbufferPSRegCounter++;

	RENDER_TEXTURE_DESC renderTextureDesc;
	renderTextureDesc.textureWidth = swapChainDesc.Width;
	renderTextureDesc.textureHeight = swapChainDesc.Height;
	renderTextureDesc.shaderType = ShaderType::PIXEL_SHADER;
	renderTextureDesc.registerSlot = m_texturePSRegCounter;
	m_pShadowMap = D3D11RenderTexture::Create(device, renderTextureDesc);
	m_texturePSRegCounter++;

	//m_pShadowMap->SetShaderResource(context);

	D3D11_SAMPLER_DESC samplerDesc;
	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.MipLODBias = 0.0f;
	samplerDesc.MaxAnisotropy = 1;
	samplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
	samplerDesc.BorderColor[0] = 0;
	samplerDesc.BorderColor[1] = 0;
	samplerDesc.BorderColor[2] = 0;
	samplerDesc.BorderColor[3] = 0;
	samplerDesc.MinLOD = 0;
	samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

	DX::ThrowIfFailed(
		device.CreateSamplerState(&samplerDesc, &m_pSampleStateWrap));

	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;

	DX::ThrowIfFailed(
		device.CreateSamplerState(&samplerDesc, &m_pSampleStateClamp));
}

void D3D11PipelineState::Shutdown()
{
	SAFE_RELEASE(m_pVS);
	SAFE_RELEASE(m_pPS);
	SAFE_RELEASE(m_pIL);
	SAFE_RELEASE(m_pRasterizerState);
	SAFE_RELEASE(m_pDepthStencilState);

	SAFE_RELEASE(m_pSampleStateClamp);
	SAFE_RELEASE(m_pSampleStateWrap);
	SAFE_RELEASE(m_pDepthStencilView);

	SAFE_DESTROY(m_pPerFrameCBufferVS);
	SAFE_DESTROY(m_pPerDrawCallCBufferVS);
	SAFE_DESTROY(m_pPerFrameCBufferPS);
	SAFE_DESTROY(m_pPerDrawCallCBufferPS);
}

void D3D11PipelineState::SetShadowMapRender(ID3D11DeviceContext& deviceContext)
{
	m_pShadowMap->ClearRenderTarget(deviceContext, *m_pDepthStencilView);
	m_pShadowMap->SetRenderTarget(deviceContext, *m_pDepthStencilView);
}

void D3D11PipelineState::SetBackBufferRender(ID3D11DeviceContext& deviceContext)
{
	deviceContext.ClearRenderTargetView(m_pRenderTargetView, m_clearColor);
	deviceContext.ClearDepthStencilView(m_pDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);
	deviceContext.OMSetRenderTargets(1,
		&m_pRenderTargetView,
		m_pDepthStencilView);
}

void D3D11PipelineState::UpdateVSPerFrame(PerFrameDataVS& data)
{
	m_perFrameDataVS.lightPos = data.lightPos;
	m_perFrameDataVS.cameraViewMatrix = data.cameraViewMatrix;
	m_perFrameDataVS.cameraProjMatrix = data.cameraProjMatrix;
	m_perFrameDataVS.lightViewMatrix = data.lightViewMatrix;
	m_perFrameDataVS.lightProjMatrix = data.lightProjMatrix;
}

void D3D11PipelineState::UpdateVSPerDrawCall(PerDrawCallDataVS& data)
{
	m_perDrawCallDataVS.worldMatrix = data.worldMatrix;
}

void D3D11PipelineState::UpdatePSPerFrame(PerFrameDataPS& data)
{
	m_perFrameDataPS.ambientColor = data.ambientColor;
	m_perFrameDataPS.diffuseColor = data.diffuseColor;
}

void D3D11PipelineState::UpdatePSPerDrawCall(PerDrawCallDataPS& data)
{
	m_perDrawCallDataPS.surfaceColor = data.surfaceColor;
}

void D3D11PipelineState::Bind(ID3D11DeviceContext& deviceContext)
{
	deviceContext.IASetInputLayout(m_pIL);
	deviceContext.VSSetShader(m_pVS, nullptr, 0);
	deviceContext.PSSetShader(m_pPS, nullptr, 0);
	
	//deviceContext.PSSetSamplers(0, 1, &m_pSampleStateClamp);
	//deviceContext.PSSetSamplers(1, 1, &m_pSampleStateWrap);
	//deviceContext.OMSetDepthStencilState(m_pDepthStencilState, 1);
	deviceContext.RSSetState(m_pRasterizerState);
}

void D3D11PipelineState::BindConstantBuffers(ID3D11DeviceContext& deviceContext)
{
	m_pPerFrameCBufferVS->Bind(deviceContext, &m_perFrameDataVS);
	m_pPerDrawCallCBufferVS->Bind(deviceContext, &m_perDrawCallDataVS);
	m_pPerFrameCBufferPS->Bind(deviceContext, &m_perFrameDataPS);
	m_pPerDrawCallCBufferPS->Bind(deviceContext, &m_perDrawCallDataPS);
}