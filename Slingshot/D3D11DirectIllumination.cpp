#include "D3D11DirectIllumination.h"

D3D11DirectIllumination* D3D11DirectIllumination::Create(D3D11Context& context)
{
	return new D3D11DirectIllumination(context);
}

D3D11DirectIllumination::D3D11DirectIllumination(D3D11Context& context) :
	m_cbufferVSRegCounter(0), m_cbufferPSRegCounter(0), m_samplerPSRegCounter(0)
{
	m_clearColor[0] = 0.0f;
	m_clearColor[1] = 0.0f;
	m_clearColor[2] = 0.0f;
	m_clearColor[3] = 1.0f;

	DXGI_SWAP_CHAIN_DESC1 swapChainDesc;
	context.GetSwapChain()->GetDesc1(&swapChainDesc);

	ZeroMemory(&m_viewport, sizeof(D3D11_VIEWPORT));
	m_viewport.Width = static_cast<float>(swapChainDesc.Width);
	m_viewport.Height = static_cast<float>(swapChainDesc.Height);
	m_viewport.MinDepth = 0.0f;
	m_viewport.MaxDepth = 1.0f;
	m_viewport.TopLeftX = 0.0f;
	m_viewport.TopLeftY = 0.0f;

	char* bytecodeVS = nullptr, * bytecodePS = nullptr;
	size_t sizeVS, sizePS;

	bytecodeVS = GetBytecode("GoochIlluminationVS.cso", sizeVS);
	bytecodePS = GetBytecode("GoochIlluminationPS.cso", sizePS);

	context.GetDevice()->CreateVertexShader(bytecodeVS, sizeVS, nullptr, &m_pVS);
	context.GetDevice()->CreatePixelShader(bytecodePS, sizePS, nullptr, &m_pPS);

	D3D11_INPUT_ELEMENT_DESC inputDesc[3];

	inputDesc[0].SemanticName = "POSITION";
	inputDesc[0].SemanticIndex = 0;
	inputDesc[0].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
	inputDesc[0].InputSlot = 0;
	inputDesc[0].AlignedByteOffset = 0;
	inputDesc[0].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
	inputDesc[0].InstanceDataStepRate = 0;

	inputDesc[1].SemanticName = "NORMAL";
	inputDesc[1].SemanticIndex = 0;
	inputDesc[1].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
	inputDesc[1].InputSlot = 0;
	inputDesc[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
	inputDesc[1].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
	inputDesc[1].InstanceDataStepRate = 0;

	inputDesc[2].SemanticName = "TEXCOORD";
	inputDesc[2].SemanticIndex = 0;
	inputDesc[2].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32_FLOAT;
	inputDesc[2].InputSlot = 0;
	inputDesc[2].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
	inputDesc[2].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
	inputDesc[2].InstanceDataStepRate = 0;

	context.GetDevice()->CreateInputLayout(inputDesc, 3, bytecodeVS, sizeVS, &m_pIL);

	SAFE_DELETE_ARRAY(bytecodeVS);
	SAFE_DELETE_ARRAY(bytecodePS);

	ID3D11Texture2D* pBackBuffer;
	DX::ThrowIfFailed(context.GetSwapChain()->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer)));
	if (pBackBuffer != nullptr) {
		DX::ThrowIfFailed(context.GetDevice()->CreateRenderTargetView(pBackBuffer, nullptr, &m_pRenderTargetView));
		SAFE_RELEASE(pBackBuffer);
	}

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
	DX::ThrowIfFailed(
		context.GetDevice()->CreateTexture2D(&depthStencilBufferDesc, nullptr, &depthStencilBuffer));

	D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
	ZeroMemory(&depthStencilViewDesc, sizeof(depthStencilViewDesc));
	depthStencilViewDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	depthStencilViewDesc.Texture2D.MipSlice = 0;

	DX::ThrowIfFailed(
		context.GetDevice()->CreateDepthStencilView(
			depthStencilBuffer,
			&depthStencilViewDesc,
			&m_pDepthStencilView));

	SAFE_RELEASE(depthStencilBuffer);

	D3D11_SAMPLER_DESC samplerDesc;
	ZeroMemory(&samplerDesc, sizeof(samplerDesc));
	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;//D3D11_FILTER_COMPARISON_MIN_MAG_MIP_POINT; D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.MipLODBias = 0.0f;
	samplerDesc.MaxAnisotropy = 1;
	samplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS; //D3D11_COMPARISON_LESS_EQUAL;
	samplerDesc.BorderColor[0] = 0.0f;
	samplerDesc.BorderColor[1] = 0.0f;
	samplerDesc.BorderColor[2] = 0.0f;
	samplerDesc.BorderColor[3] = 0.0f;
	samplerDesc.MinLOD = 0.0f;
	samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

	DX::ThrowIfFailed(
		context.GetDevice()->CreateSamplerState(&samplerDesc, &m_pShadowMapSamplerState));

	D3D11_RASTERIZER_DESC rsStateDesc;
	ZeroMemory(&rsStateDesc, sizeof(rsStateDesc));
	rsStateDesc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
	rsStateDesc.CullMode = D3D11_CULL_MODE::D3D11_CULL_BACK;
	rsStateDesc.FrontCounterClockwise = false;
	rsStateDesc.DepthBias = 0;
	rsStateDesc.DepthBiasClamp = 0.0f;
	rsStateDesc.SlopeScaledDepthBias = 0.0f;
	rsStateDesc.DepthClipEnable = true;
	rsStateDesc.ScissorEnable = false;
	rsStateDesc.MultisampleEnable = false;
	rsStateDesc.AntialiasedLineEnable = false;

	DX::ThrowIfFailed(
		context.GetDevice()->CreateRasterizerState(&rsStateDesc, &m_pRasterizerState));

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
	DX::ThrowIfFailed(
		context.GetDevice()->CreateDepthStencilState(&depthStencilStateDesc, &m_pDepthStencilState));

	CONSTANT_BUFFER_DESC desc0;
	desc0.cbufferData = &m_perFrameDataVS;
	desc0.cbufferSize = sizeof(m_perFrameDataVS);
	desc0.shaderType = ShaderType::VERTEX_SHADER;
	desc0.registerSlot = m_cbufferVSRegCounter;
	m_pPerFrameCBufferVS = D3D11ConstantBuffer::Create(*context.GetDevice(), desc0);
	m_cbufferVSRegCounter++;

	CONSTANT_BUFFER_DESC desc1;
	desc1.cbufferData = &m_perDrawCallDataVS;
	desc1.cbufferSize = sizeof(m_perDrawCallDataVS);
	desc1.shaderType = ShaderType::VERTEX_SHADER;
	desc1.registerSlot = m_cbufferVSRegCounter;
	m_pPerDrawCallCBufferVS = D3D11ConstantBuffer::Create(*context.GetDevice(), desc1);
	m_cbufferVSRegCounter++;

	CONSTANT_BUFFER_DESC desc2;
	desc2.cbufferData = &m_perFrameDataPS;
	desc2.cbufferSize = sizeof(m_perFrameDataPS);
	desc2.shaderType = ShaderType::PIXEL_SHADER;
	desc2.registerSlot = m_cbufferPSRegCounter;
	m_pPerFrameCBufferPS = D3D11ConstantBuffer::Create(*context.GetDevice(), desc2);
	m_cbufferPSRegCounter++;

	CONSTANT_BUFFER_DESC desc3;
	desc3.cbufferData = &m_perDrawCallDataPS;
	desc3.cbufferSize = sizeof(m_perDrawCallDataPS);
	desc3.shaderType = ShaderType::PIXEL_SHADER;
	desc3.registerSlot = m_cbufferPSRegCounter;
	m_pPerDrawCallCBufferPS = D3D11ConstantBuffer::Create(*context.GetDevice(), desc3);
	m_cbufferPSRegCounter++;
}

void D3D11DirectIllumination::Shutdown()
{
	SAFE_RELEASE(m_pIL);
	SAFE_RELEASE(m_pVS);
	SAFE_RELEASE(m_pPS);

	SAFE_RELEASE(m_pDepthStencilState);
	SAFE_RELEASE(m_pRasterizerState);

	SAFE_RELEASE(m_pRenderTargetView);
	SAFE_RELEASE(m_pDepthStencilView);

	SAFE_RELEASE(m_pShadowMapSamplerState);

	SAFE_DESTROY(m_pPerFrameCBufferVS);
	SAFE_DESTROY(m_pPerFrameCBufferPS);
	SAFE_DESTROY(m_pPerDrawCallCBufferVS);
	SAFE_DESTROY(m_pPerDrawCallCBufferPS);
}

void D3D11DirectIllumination::UpdatePerConfig(ID3D11DeviceContext& deviceContext)
{
}

void D3D11DirectIllumination::UpdatePerFrame(ID3D11DeviceContext& deviceContext, ID3D11ShaderResourceView* depthMap)
{
	deviceContext.IASetInputLayout(m_pIL);
	deviceContext.VSSetShader(m_pVS, nullptr, 0);
	deviceContext.GSSetShader(nullptr, nullptr, 0);
	deviceContext.PSSetShader(m_pPS, nullptr, 0);
	deviceContext.PSSetSamplers(0, 1, &m_pShadowMapSamplerState);
	deviceContext.RSSetState(m_pRasterizerState);
	deviceContext.RSSetViewports(1, &m_viewport);
	//deviceContext.OMSetDepthStencilState(m_pDepthStencilState, 1);

	deviceContext.ClearRenderTargetView(m_pRenderTargetView, m_clearColor);
	deviceContext.ClearDepthStencilView(m_pDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);
	deviceContext.OMSetRenderTargets(1,
		&m_pRenderTargetView,
		m_pDepthStencilView);

	deviceContext.PSSetShaderResources(0, 1, &depthMap);
}

void D3D11DirectIllumination::UpdateBuffersPerFrame(PerFrameDataVS_DI& dataVS, PerFrameDataPS_DI& dataPS)
{
	m_perFrameDataVS = dataVS;
	m_perFrameDataPS = dataPS;
}

void D3D11DirectIllumination::UpdateBuffersPerDrawCall(PerDrawCallDataVS_DI& dataVS, PerDrawCallDataPS_DI& dataPS)
{
	m_perDrawCallDataVS = dataVS;
	m_perDrawCallDataPS = dataPS;
}

void D3D11DirectIllumination::BindConstantBuffers(ID3D11DeviceContext& deviceContext)
{
	m_pPerFrameCBufferVS->Bind(deviceContext, &m_perFrameDataVS);
	m_pPerDrawCallCBufferVS->Bind(deviceContext, &m_perDrawCallDataVS);
	m_pPerFrameCBufferPS->Bind(deviceContext, &m_perFrameDataPS);
	m_pPerDrawCallCBufferPS->Bind(deviceContext, &m_perDrawCallDataPS);
}

void D3D11DirectIllumination::EndFrameRender(ID3D11DeviceContext& deviceContext)
{
	ID3D11ShaderResourceView* nullSRV[1] = { nullptr };
	deviceContext.PSSetShaderResources(0, 1, nullSRV);
}
