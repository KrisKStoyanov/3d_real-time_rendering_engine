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
	m_pVS_DirectIllumination(nullptr), m_pPS_DirectIllumination(nullptr), m_pIL_DirectIllumination(nullptr), 
	m_pVS_DepthMap(nullptr), m_pPS_DepthMap(nullptr), m_pIL_DepthMap(nullptr),
	m_cbufferVSRegCounter_DI(0), m_cbufferPSRegCounter_DI(0),
	m_samplerVSRegCounter_DI(0), m_samplerPSRegCounter_DI(0),
	m_textureVSRegCounter_DI(0), m_texturePSRegCounter_DI(0),
	m_cbufferVSRegCounter_DM(0),
	m_perFrameDataVS_DI(), m_perDrawCallDataVS_DI(), m_perFrameDataPS_DI(), m_perDrawCallDataPS_DI(),
	m_perFrameDataVS_DM(), m_perDrawCallDataVS_DM(),
	m_pPerFrameCBufferVS_DI(nullptr), m_pPerDrawCallCBufferVS_DI(nullptr),
	m_pPerFrameCBufferPS_DI(nullptr), m_pPerDrawCallCBufferPS_DI(nullptr),
	m_pPerFrameCBufferVS_DM(nullptr), m_pPerDrawCallCBufferVS_DM(nullptr),
	 m_pSampleStateWrap(nullptr), m_pDepthStencilState(nullptr), m_pDepthStencilView(nullptr),
	m_pRasterizerState(nullptr), m_pRenderTargetView(nullptr)
{

	m_clearColor[0] = 0.0f;
	m_clearColor[1] = 0.0f;
	m_clearColor[2] = 0.0f;
	m_clearColor[3] = 1.0f;

	char* VS_bytecode_DI = nullptr, * PS_bytecode_DI = nullptr;
	size_t VS_size_DI, PS_size_DI;
	VS_bytecode_DI = GetFileBytecode(desc.VS_filename_DI, VS_size_DI);
	PS_bytecode_DI = GetFileBytecode(desc.PS_filename_DI, PS_size_DI);

	device.CreateVertexShader(VS_bytecode_DI, VS_size_DI, nullptr, &m_pVS_DirectIllumination);
	device.CreatePixelShader(PS_bytecode_DI, PS_size_DI, nullptr, &m_pPS_DirectIllumination);

	D3D11_INPUT_ELEMENT_DESC IL_desc_DI[3];

	IL_desc_DI[0].SemanticName = "POSITION";
	IL_desc_DI[0].SemanticIndex = 0;
	IL_desc_DI[0].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
	IL_desc_DI[0].InputSlot = 0;
	IL_desc_DI[0].AlignedByteOffset = 0;
	IL_desc_DI[0].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
	IL_desc_DI[0].InstanceDataStepRate = 0;

	IL_desc_DI[1].SemanticName = "NORMAL";
	IL_desc_DI[1].SemanticIndex = 0;
	IL_desc_DI[1].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
	IL_desc_DI[1].InputSlot = 0;
	IL_desc_DI[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
	IL_desc_DI[1].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
	IL_desc_DI[1].InstanceDataStepRate = 0;

	IL_desc_DI[2].SemanticName = "TEXCOORD";
	IL_desc_DI[2].SemanticIndex = 0;
	IL_desc_DI[2].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32_FLOAT;
	IL_desc_DI[2].InputSlot = 0;
	IL_desc_DI[2].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
	IL_desc_DI[2].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
	IL_desc_DI[2].InstanceDataStepRate = 0;

	device.CreateInputLayout(IL_desc_DI, 3, VS_bytecode_DI, VS_size_DI, &m_pIL_DirectIllumination);

	SAFE_DELETE_ARRAY(VS_bytecode_DI);
	SAFE_DELETE_ARRAY(PS_bytecode_DI);

	//-----------

	char* VS_bytecode_DM = nullptr, * PS_bytecode_DM = nullptr;
	size_t VS_size_DM, PS_size_DM;
	VS_bytecode_DM = GetFileBytecode(desc.VS_filename_DM, VS_size_DM);
	PS_bytecode_DM = GetFileBytecode(desc.PS_filename_DM, PS_size_DM);

	device.CreateVertexShader(VS_bytecode_DM, VS_size_DM, nullptr, &m_pVS_DepthMap);
	device.CreatePixelShader(PS_bytecode_DM, PS_size_DM, nullptr, &m_pPS_DepthMap);

	D3D11_INPUT_ELEMENT_DESC IL_desc_DM[1];

	IL_desc_DM[0].SemanticName = "POSITION";
	IL_desc_DM[0].SemanticIndex = 0;
	IL_desc_DM[0].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32_FLOAT;
	IL_desc_DM[0].InputSlot = 0;
	IL_desc_DM[0].AlignedByteOffset = 0;
	IL_desc_DM[0].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
	IL_desc_DM[0].InstanceDataStepRate = 0;

	device.CreateInputLayout(IL_desc_DM, 1, VS_bytecode_DM, VS_size_DM, &m_pIL_DepthMap);

	SAFE_DELETE_ARRAY(VS_bytecode_DM);
	SAFE_DELETE_ARRAY(PS_bytecode_DM);

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
	desc0.cbufferData = &m_perFrameDataVS_DI;
	desc0.cbufferSize = sizeof(m_perFrameDataVS_DI);
	desc0.shaderType = ShaderType::VERTEX_SHADER;
	desc0.registerSlot = m_cbufferVSRegCounter_DI;
	m_pPerFrameCBufferVS_DI = D3D11ConstantBuffer::Create(device, desc0);
	m_cbufferVSRegCounter_DI++;

	CONSTANT_BUFFER_DESC desc1;
	desc1.cbufferData = &m_perDrawCallDataVS_DI;
	desc1.cbufferSize = sizeof(m_perDrawCallDataVS_DI);
	desc1.shaderType = ShaderType::VERTEX_SHADER;
	desc1.registerSlot = m_cbufferVSRegCounter_DI;
	m_pPerDrawCallCBufferVS_DI = D3D11ConstantBuffer::Create(device, desc1);
	m_cbufferVSRegCounter_DI++;

	CONSTANT_BUFFER_DESC desc2;
	desc2.cbufferData = &m_perFrameDataPS_DI;
	desc2.cbufferSize = sizeof(m_perFrameDataPS_DI);
	desc2.shaderType = ShaderType::PIXEL_SHADER;
	desc2.registerSlot = m_cbufferPSRegCounter_DI;
	m_pPerFrameCBufferPS_DI = D3D11ConstantBuffer::Create(device, desc2);
	m_cbufferPSRegCounter_DI++;

	CONSTANT_BUFFER_DESC desc3;
	desc3.cbufferData = &m_perDrawCallDataPS_DI;
	desc3.cbufferSize = sizeof(m_perDrawCallDataPS_DI);
	desc3.shaderType = ShaderType::PIXEL_SHADER;
	desc3.registerSlot = m_cbufferPSRegCounter_DI;
	m_pPerDrawCallCBufferPS_DI = D3D11ConstantBuffer::Create(device, desc3);
	m_cbufferPSRegCounter_DI++;

	CONSTANT_BUFFER_DESC desc4;
	desc4.cbufferData = &m_perFrameDataVS_DM;
	desc4.cbufferSize = sizeof(m_perFrameDataVS_DM);
	desc4.shaderType = ShaderType::VERTEX_SHADER;
	desc4.registerSlot = m_cbufferVSRegCounter_DM;
	m_pPerFrameCBufferVS_DM = D3D11ConstantBuffer::Create(device, desc4);
	m_cbufferVSRegCounter_DM++;

	CONSTANT_BUFFER_DESC desc5;
	desc5.cbufferData = &m_perDrawCallDataVS_DM;
	desc5.cbufferSize = sizeof(m_perDrawCallDataVS_DM);
	desc5.shaderType = ShaderType::VERTEX_SHADER;
	desc5.registerSlot = m_cbufferVSRegCounter_DM;
	m_pPerDrawCallCBufferVS_DM = D3D11ConstantBuffer::Create(device, desc5);
	m_cbufferVSRegCounter_DM++;

	RENDER_TEXTURE_DESC renderTextureDesc;
	renderTextureDesc.textureWidth = swapChainDesc.Width;
	renderTextureDesc.textureHeight = swapChainDesc.Height;
	renderTextureDesc.shaderType = ShaderType::PIXEL_SHADER;
	renderTextureDesc.registerSlot = m_texturePSRegCounter_DI;
	m_pDepthMap = D3D11RenderTexture::Create(device, renderTextureDesc);
	m_texturePSRegCounter_DI++;

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
}

void D3D11PipelineState::Shutdown()
{
	SAFE_RELEASE(m_pVS_DirectIllumination);
	SAFE_RELEASE(m_pPS_DirectIllumination);
	SAFE_RELEASE(m_pIL_DirectIllumination);

	SAFE_RELEASE(m_pVS_DepthMap);
	SAFE_RELEASE(m_pPS_DepthMap);
	SAFE_RELEASE(m_pIL_DepthMap);

	SAFE_RELEASE(m_pRasterizerState);
	SAFE_RELEASE(m_pDepthStencilState);

	SAFE_RELEASE(m_pSampleStateWrap);
	SAFE_RELEASE(m_pDepthStencilView);

	SAFE_DESTROY(m_pPerFrameCBufferVS_DI);
	SAFE_DESTROY(m_pPerDrawCallCBufferVS_DI);
	SAFE_DESTROY(m_pPerFrameCBufferPS_DI);
	SAFE_DESTROY(m_pPerDrawCallCBufferPS_DI);
}

void D3D11PipelineState::SetDepthMapRender(ID3D11DeviceContext& deviceContext)
{
	m_pDepthMap->ClearRenderTarget(deviceContext, *m_pDepthStencilView);
	m_pDepthMap->SetRenderTarget(deviceContext, *m_pDepthStencilView);
}

void D3D11PipelineState::SetBackBufferRender(ID3D11DeviceContext& deviceContext)
{
	deviceContext.ClearRenderTargetView(m_pRenderTargetView, m_clearColor);
	deviceContext.ClearDepthStencilView(m_pDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);
	deviceContext.OMSetRenderTargets(1,
		&m_pRenderTargetView,
		m_pDepthStencilView);
}

void D3D11PipelineState::UpdatePerConfig(ID3D11DeviceContext& deviceContext)
{
	//deviceContext.IASetInputLayout(m_pIL);
	//deviceContext.VSSetShader(m_pVS, nullptr, 0);
	//deviceContext.PSSetShader(m_pPS, nullptr, 0);
	//m_pShadowMap->SetShaderResource(deviceContext);
	//deviceContext.PSSetSamplers(0, 1, &m_pSampleStateWrap);
	//deviceContext.OMSetDepthStencilState(m_pDepthStencilState, 1);
	//deviceContext.RSSetState(m_pRasterizerState);
}

void D3D11PipelineState::UpdatePerFrame_DM(ID3D11DeviceContext& deviceContext)
{
	deviceContext.IASetInputLayout(m_pIL_DepthMap);
	deviceContext.VSSetShader(m_pVS_DepthMap, nullptr, 0);
	deviceContext.PSSetShader(m_pPS_DepthMap, nullptr, 0);
	deviceContext.OMSetDepthStencilState(m_pDepthStencilState, 1);
	deviceContext.RSSetState(m_pRasterizerState);
}

void D3D11PipelineState::UpdatePerFrame_DI(ID3D11DeviceContext& deviceContext)
{
	deviceContext.IASetInputLayout(m_pIL_DirectIllumination);
	deviceContext.VSSetShader(m_pVS_DirectIllumination, nullptr, 0);
	deviceContext.PSSetShader(m_pPS_DirectIllumination, nullptr, 0);
	deviceContext.PSSetSamplers(0, 1, &m_pSampleStateWrap);
	deviceContext.OMSetDepthStencilState(m_pDepthStencilState, 1);
	deviceContext.RSSetState(m_pRasterizerState);
}

void D3D11PipelineState::BindShaderResources(ID3D11DeviceContext& deviceContext)
{
	m_pDepthMap->SetShaderResource(deviceContext);
}

void D3D11PipelineState::UnbindShaderResources(ID3D11DeviceContext& deviceContext)
{
	m_pDepthMap->UnsetShaderResource(deviceContext);
}

void D3D11PipelineState::UpdateVSPerFrame_DM(PerFrameDataVS_DM& data)
{
	m_perFrameDataVS_DM = data;
}

void D3D11PipelineState::UpdateVSPerDrawCall_DM(PerDrawCallDataVS_DM& data)
{
	m_perDrawCallDataVS_DM = data;
}

void D3D11PipelineState::UpdateVSPerFrame_DI(PerFrameDataVS_DI& data)
{
	m_perFrameDataVS_DI.lightPos = data.lightPos;
	m_perFrameDataVS_DI.cameraViewMatrix = data.cameraViewMatrix;
	m_perFrameDataVS_DI.cameraProjMatrix = data.cameraProjMatrix;
	m_perFrameDataVS_DI.lightViewMatrix = data.lightViewMatrix;
	m_perFrameDataVS_DI.lightProjMatrix = data.lightProjMatrix;
}

void D3D11PipelineState::UpdateVSPerDrawCall_DI(PerDrawCallDataVS_DI& data)
{
	m_perDrawCallDataVS_DI.worldMatrix = data.worldMatrix;
}

void D3D11PipelineState::UpdatePSPerFrame_DI(PerFrameDataPS_DI& data)
{
	m_perFrameDataPS_DI.ambientColor = data.ambientColor;
	m_perFrameDataPS_DI.diffuseColor = data.diffuseColor;
}

void D3D11PipelineState::UpdatePSPerDrawCall_DI(PerDrawCallDataPS_DI& data)
{
	m_perDrawCallDataPS_DI.surfaceColor = data.surfaceColor;
}

void D3D11PipelineState::BindConstantBuffers_DI(ID3D11DeviceContext& deviceContext)
{
	m_pPerFrameCBufferVS_DI->Bind(deviceContext, &m_perFrameDataVS_DI);
	m_pPerDrawCallCBufferVS_DI->Bind(deviceContext, &m_perDrawCallDataVS_DI);
	m_pPerFrameCBufferPS_DI->Bind(deviceContext, &m_perFrameDataPS_DI);
	m_pPerDrawCallCBufferPS_DI->Bind(deviceContext, &m_perDrawCallDataPS_DI);
}

void D3D11PipelineState::BindConstantBuffers_DM(ID3D11DeviceContext& deviceContext)
{
	m_pPerFrameCBufferVS_DM->Bind(deviceContext, &m_perFrameDataVS_DM);
	m_pPerDrawCallCBufferVS_DM->Bind(deviceContext, &m_perDrawCallDataVS_DM);
}