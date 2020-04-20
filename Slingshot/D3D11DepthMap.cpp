#include "D3D11DepthMap.h"

D3D11DepthMap* D3D11DepthMap::Create(D3D11Context& context)
{
	return new D3D11DepthMap(context);
}

D3D11DepthMap::D3D11DepthMap(D3D11Context& context) :
	m_cbufferVSRegCounter(0), m_cbufferGSRegCounter(0)
{
	UINT shadowMapWidth = 800, shadowMapHeight = 800;
	ZeroMemory(&m_viewport, sizeof(D3D11_VIEWPORT));
	m_viewport.Width = static_cast<float>(shadowMapWidth);
	m_viewport.Height = static_cast<float>(shadowMapHeight);
	m_viewport.MinDepth = 0.0f;
	m_viewport.MaxDepth = 1.0f;

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

	ID3D11Texture2D* shadowMap;

	D3D11_TEXTURE2D_DESC smDesc;
	ZeroMemory(&smDesc, sizeof(smDesc));
	smDesc.Width = shadowMapWidth;
	smDesc.Height = shadowMapHeight;
	smDesc.MipLevels = 1;
	smDesc.ArraySize = 1; //6;
	smDesc.Format = DXGI_FORMAT_R24G8_TYPELESS;
	smDesc.SampleDesc.Count = 1;
	smDesc.SampleDesc.Quality = 0;
	smDesc.Usage = D3D11_USAGE_DEFAULT;
	smDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
	smDesc.CPUAccessFlags = 0;
	smDesc.MiscFlags = 0; //D3D11_RESOURCE_MISC_TEXTURECUBE;
	DX::ThrowIfFailed(
		context.GetDevice()->CreateTexture2D(
			&smDesc, 
			nullptr, 
			&shadowMap));

	D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;
	ZeroMemory(&dsvDesc, sizeof(dsvDesc));
	dsvDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D; //D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
	dsvDesc.Texture2D.MipSlice = 0;
	//dsvDesc.Texture2DArray.ArraySize = 6;
	//dsvDesc.Texture2DArray.FirstArraySlice = 0;
	//dsvDesc.Texture2DArray.MipSlice = 0;
	DX::ThrowIfFailed(
		context.GetDevice()->CreateDepthStencilView(
			shadowMap,
			&dsvDesc,
			&m_pShadowMapDSV));

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	ZeroMemory(&srvDesc, sizeof(srvDesc));
	srvDesc.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D; //D3D11_SRV_DIMENSION_TEXTURECUBE;
	srvDesc.Texture2D.MipLevels = 1;
	srvDesc.Texture2D.MostDetailedMip = 0;
	//srvDesc.Texture2DArray.ArraySize = 6;
	//srvDesc.Texture2DArray.FirstArraySlice = 0;
	//srvDesc.Texture2DArray.MostDetailedMip = 0;
	//srvDesc.Texture2DArray.MipLevels = 1;
	DX::ThrowIfFailed(
		context.GetDevice()->CreateShaderResourceView(
			shadowMap, 
			&srvDesc, 
			&m_pShadowMapSRV));

	SAFE_RELEASE(shadowMap);

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

	context.GetDevice()->CreateRasterizerState(&rsStateDesc, &m_pRasterizerState);

	//CONSTANT_BUFFER_DESC desc0;
	//desc0.cbufferData = &m_perFrameDataGS;
	//desc0.cbufferSize = sizeof(m_perFrameDataGS);
	//desc0.shaderType = ShaderType::GEOMETRY_SHADER;
	//desc0.registerSlot = m_cbufferGSRegCounter;
	//m_pPerFrameCBufferGS = D3D11ConstantBuffer::Create(*context.GetDevice(), desc0);
	//m_cbufferGSRegCounter++;

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
}

void D3D11DepthMap::Shutdown()
{
	SAFE_RELEASE(m_pIL);
	SAFE_RELEASE(m_pVS);
	SAFE_RELEASE(m_pGS);
	SAFE_RELEASE(m_pPS);

	SAFE_RELEASE(m_pRasterizerState);

	SAFE_DESTROY(m_pPerFrameCBufferGS);
	SAFE_DESTROY(m_pPerFrameCBufferVS);
	SAFE_DESTROY(m_pPerDrawCallCBufferVS);
}

void D3D11DepthMap::UpdatePerConfig(ID3D11DeviceContext& deviceContext)
{

}

void D3D11DepthMap::UpdatePerFrame(ID3D11DeviceContext& deviceContext)
{
	deviceContext.IASetInputLayout(m_pIL);
	deviceContext.VSSetShader(m_pVS, nullptr, 0);
	//deviceContext.GSSetShader(m_pGS, nullptr, 0);
	deviceContext.PSSetShader(m_pPS, nullptr, 0);
	deviceContext.RSSetState(m_pRasterizerState);
	deviceContext.RSSetViewports(1, &m_viewport);

	deviceContext.ClearDepthStencilView(m_pShadowMapDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
	deviceContext.OMSetRenderTargets(0,
		nullptr,
		m_pShadowMapDSV);

	deviceContext.PSSetShaderResources(0, 0, nullptr);
}

void D3D11DepthMap::UpdateBuffersPerFrame(PerFrameDataVS_DM& data)
{
	m_perFrameDataVS = data;
}

void D3D11DepthMap::UpdateBuffersPerDrawCall(PerDrawCallDataVS_DM& data)
{
	m_perDrawCallDataVS = data;
}

void D3D11DepthMap::BindConstantBuffers(ID3D11DeviceContext& deviceContext)
{
	m_pPerDrawCallCBufferVS->Bind(deviceContext, &m_perDrawCallDataVS);
	m_pPerFrameCBufferVS->Bind(deviceContext, &m_perFrameDataVS);
	//m_pPerFrameCBufferGS->Bind(deviceContext, &m_perFrameDataGS);
}
