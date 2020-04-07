#include "D3D11DepthMap.h"

D3D11DepthMap* D3D11DepthMap::Create(D3D11Context& context)
{
	return new D3D11DepthMap(context);
}

D3D11DepthMap::D3D11DepthMap(D3D11Context& context) :
	m_cbufferVSRegCounter(0), m_cbufferGSRegCounter(0)
{
	char* bytecodeVS = nullptr, *bytecodeGS = nullptr, * bytecodePS = nullptr;
	size_t sizeVS, sizeGS, sizePS;

	bytecodeVS = GetBytecode("DepthMapVS.cso", sizeVS);
	bytecodeGS = GetBytecode("DepthMapGS.cso", sizeGS);
	bytecodePS = GetBytecode("DepthMapPS.cso", sizePS);

	context.GetDevice()->CreateVertexShader(bytecodeVS, sizeVS, nullptr, &m_pVS);
	context.GetDevice()->CreateGeometryShader(bytecodeGS, sizeGS, nullptr, &m_pGS);
	context.GetDevice()->CreatePixelShader(bytecodePS, sizePS, nullptr, &m_pPS);

	D3D11_INPUT_ELEMENT_DESC inputDesc[1];

	inputDesc[0].SemanticName = "POSITION";
	inputDesc[0].SemanticIndex = 0;
	inputDesc[0].Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32_FLOAT;
	inputDesc[0].InputSlot = 0;
	inputDesc[0].AlignedByteOffset = 0;
	inputDesc[0].InputSlotClass = D3D11_INPUT_CLASSIFICATION::D3D11_INPUT_PER_VERTEX_DATA;
	inputDesc[0].InstanceDataStepRate = 0;

	context.GetDevice()->CreateInputLayout(inputDesc, 1, bytecodeVS, sizeVS, &m_pIL);

	SAFE_DELETE_ARRAY(bytecodeVS);
	SAFE_DELETE_ARRAY(bytecodeGS);
	SAFE_DELETE_ARRAY(bytecodePS);

	DXGI_SWAP_CHAIN_DESC1 swapChainDesc;
	context.GetSwapChain()->GetDesc1(&swapChainDesc);

	for (int i = 0; i < 6; ++i)
	{
		ID3D11Texture2D* renderTexture;

		D3D11_TEXTURE2D_DESC textureDesc;
		ZeroMemory(&textureDesc, sizeof(textureDesc));
		textureDesc.Width = swapChainDesc.Width;
		textureDesc.Height = swapChainDesc.Height;
		textureDesc.MipLevels = 1;
		textureDesc.ArraySize = 1;
		textureDesc.Format = DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT;
		textureDesc.SampleDesc.Count = 1;
		textureDesc.Usage = D3D11_USAGE_DEFAULT;
		textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
		textureDesc.CPUAccessFlags = 0;
		textureDesc.MiscFlags = 0;
		DX::ThrowIfFailed(
			context.GetDevice()->CreateTexture2D(&textureDesc, nullptr, &renderTexture));

		D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
		rtvDesc.Format = textureDesc.Format;
		rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
		rtvDesc.Texture2D.MipSlice = 0;
		DX::ThrowIfFailed(
			context.GetDevice()->CreateRenderTargetView(renderTexture, &rtvDesc, &m_pDepthMapsRTV[i]));

		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
		srvDesc.Format = textureDesc.Format;
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MostDetailedMip = 0;
		srvDesc.Texture2D.MipLevels = 1;
		DX::ThrowIfFailed(
			context.GetDevice()->CreateShaderResourceView(renderTexture, &srvDesc, &m_pDepthMapsSRV[i]));

		SAFE_RELEASE(renderTexture);
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

	if (depthStencilBuffer)
	{
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
	}

	D3D11_RASTERIZER_DESC rasterizer_desc;
	rasterizer_desc.FillMode = D3D11_FILL_MODE::D3D11_FILL_SOLID;
	rasterizer_desc.CullMode = D3D11_CULL_MODE::D3D11_CULL_BACK;
	rasterizer_desc.FrontCounterClockwise = false;
	rasterizer_desc.DepthBias = false;
	rasterizer_desc.DepthBiasClamp = 0;
	rasterizer_desc.SlopeScaledDepthBias = 0;
	rasterizer_desc.DepthClipEnable = true;
	rasterizer_desc.ScissorEnable = false;
	rasterizer_desc.MultisampleEnable = false;
	rasterizer_desc.AntialiasedLineEnable = false;

	context.GetDevice()->CreateRasterizerState(&rasterizer_desc, &m_pRasterizerState);

	CONSTANT_BUFFER_DESC desc0;
	desc0.cbufferData = &m_perFrameDataGS;
	desc0.cbufferSize = sizeof(m_perFrameDataGS);
	desc0.shaderType = ShaderType::GEOMETRY_SHADER;
	desc0.registerSlot = m_cbufferGSRegCounter;
	m_pPerFrameCBufferGS = D3D11ConstantBuffer::Create(*context.GetDevice(), desc0);
	m_cbufferGSRegCounter++;

	CONSTANT_BUFFER_DESC desc1;
	desc1.cbufferData = &m_perDrawCallDataVS;
	desc1.cbufferSize = sizeof(m_perDrawCallDataVS);
	desc1.shaderType = ShaderType::VERTEX_SHADER;
	desc1.registerSlot = m_cbufferVSRegCounter;
	m_pPerDrawCallCBufferVS = D3D11ConstantBuffer::Create(*context.GetDevice(), desc1);
	m_cbufferVSRegCounter++;
}

void D3D11DepthMap::Shutdown()
{
	SAFE_RELEASE(m_pIL);
	SAFE_RELEASE(m_pVS);
	SAFE_RELEASE(m_pGS);
	SAFE_RELEASE(m_pPS);

	SAFE_RELEASE(m_pDepthStencilState);
	SAFE_RELEASE(m_pRasterizerState);

	for (int i = 0; i < 6; ++i)
	{
		SAFE_RELEASE(m_pDepthMapsRTV[i]);
		SAFE_RELEASE(m_pDepthMapsSRV[i]);
	}
	SAFE_RELEASE(m_pDepthStencilView);

	SAFE_DESTROY(m_pPerFrameCBufferGS);
	SAFE_DESTROY(m_pPerDrawCallCBufferVS);
}

void D3D11DepthMap::UpdatePerConfig(ID3D11DeviceContext& deviceContext)
{

}

void D3D11DepthMap::UpdatePerFrame(ID3D11DeviceContext& deviceContext)
{
	deviceContext.IASetInputLayout(m_pIL);
	deviceContext.VSSetShader(m_pVS, nullptr, 0);
	deviceContext.GSSetShader(m_pGS, nullptr, 0);
	deviceContext.PSSetShader(m_pPS, nullptr, 0);
	deviceContext.OMSetDepthStencilState(m_pDepthStencilState, 1);
	deviceContext.RSSetState(m_pRasterizerState);

	for (int i = 0; i < 6; ++i)
	{
		deviceContext.ClearRenderTargetView(m_pDepthMapsRTV[i], m_clearColor);
	}
	deviceContext.ClearDepthStencilView(m_pDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);
	deviceContext.OMSetRenderTargets(6,
		m_pDepthMapsRTV,
		m_pDepthStencilView);
}

void D3D11DepthMap::UpdateBuffersPerFrame(PerFrameDataGS_DM& data)
{
	m_perFrameDataGS = data;
}

void D3D11DepthMap::UpdateBuffersPerDrawCall(PerDrawCallDataVS_DM& data)
{
	m_perDrawCallDataVS = data;
}

void D3D11DepthMap::BindConstantBuffers(ID3D11DeviceContext& deviceContext)
{
	m_pPerDrawCallCBufferVS->Bind(deviceContext, &m_perDrawCallDataVS);
	m_pPerFrameCBufferGS->Bind(deviceContext, &m_perFrameDataGS);
}
