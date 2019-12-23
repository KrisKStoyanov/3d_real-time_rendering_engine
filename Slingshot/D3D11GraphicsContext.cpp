#include "D3D11GraphicsContext.h"

D3D11GraphicsContext::D3D11GraphicsContext(HWND hwnd)
{
	m_hWnd = hwnd;
}

D3D11GraphicsContext::~D3D11GraphicsContext()
{
}

void D3D11GraphicsContext::OnCreate(HWND hwnd)
{
	if (SetupD3D11()) {
		if (SetupRenderingPipeline(m_EnableIndexing, m_EnableSO)) {
			SetupContext(
				m_pImmediateContext.Get(),
				m_pSwapChain.Get(),
				m_pVertexShader.GetAddressOf(),
				m_pInputLayout.GetAddressOf(),
				m_pGeometryShader.GetAddressOf(),
				m_pRasertizerState.GetAddressOf(),
				m_pPixelShader.GetAddressOf(),
				m_pDepthStencilState.GetAddressOf(),
				m_pBlendState.GetAddressOf());

			//Setup interop HPS extension
			//m_pDeviceInteropContext = new HC::D3D11DeviceInteropContext();
			//HC::InvokeRenderKernel(m_pDeviceInteropContext->m_ScreenBuffer, 1280, 720);	

			InitRenderingPipeline(m_EnableIndexing, m_EnableSO);
		}
	}
}

void D3D11GraphicsContext::OnDestroy()
{
	TerminateRenderingPipeline(m_EnableIndexing, m_EnableSO);
}

void D3D11GraphicsContext::OnPaint()
{
	Render();
}

void D3D11GraphicsContext::OnResize()
{
}

void D3D11GraphicsContext::OnLButtonDown(int pixelX, int pixelY, DWORD flags)
{
}

void D3D11GraphicsContext::OnLButtonUp()
{
}

void D3D11GraphicsContext::OnMouseMove(int pixelX, int pixelY, DWORD flags)
{
}

void D3D11GraphicsContext::CaptureCursor()
{
	if (m_CaptureCursor) {
		RECT rc;
		GetClientRect(m_hWnd, &rc);

		POINT pt = { rc.left, rc.top };
		POINT pt2 = { rc.right, rc.bottom };
		ClientToScreen(m_hWnd, &pt);
		ClientToScreen(m_hWnd, &pt2);

		SetRect(&rc, pt.x, pt.y, pt2.x, pt2.y);

		ClipCursor(&rc);
	}
	else {
		ClipCursor(NULL);
	}
}

bool D3D11GraphicsContext::SetupD3D11()
{
	CreateDevice(&m_pAdapter, &m_pDevice, &m_pImmediateContext);
	CreateSwapChain(m_pAdapter.Get(), &m_pSwapChain);
	CreateRenderTargetView(m_pDevice.Get(), m_pSwapChain.Get(), &m_pRenderTargetView);
	CreateDeferredContext(m_pDevice.Get(), &m_pDeferredContext);
	return true;
}

bool D3D11GraphicsContext::SetupContext(
	ID3D11DeviceContext* context,
	IDXGISwapChain1* swapChain,
	ID3D11VertexShader** vs,
	ID3D11InputLayout** il,
	ID3D11GeometryShader** gs,
	ID3D11RasterizerState** rs,
	ID3D11PixelShader** ps,
	ID3D11DepthStencilState** dss,
	ID3D11BlendState** bs)
{
	context->VSSetShader(*vs, nullptr, 0);
	context->IASetInputLayout(*il);
	//context->HSSetShader(*shader, nullptr, 0); //Pending implementation
	//context->DSSetShader(*shader, nullptr, 0); //Pending implementation
	context->GSSetShader(*gs, nullptr, 0);
	context->RSSetState(*rs);
	context->PSSetShader(*ps, nullptr, 0);
	context->OMSetDepthStencilState(*dss, 1);
	float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	UINT sampleMask = 0xffffffff;
	context->OMSetBlendState(*bs, blendFactor, sampleMask);

	Microsoft::WRL::ComPtr<ID3D11Texture2D> pBackBuffer;
	swapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
	D3D11_TEXTURE2D_DESC backBufferDesc = {};
	pBackBuffer->GetDesc(&backBufferDesc);

	SetContextRSViewports(context, backBufferDesc);
	SetContextRSScissorRect(context, backBufferDesc);

	return true;
}

bool D3D11GraphicsContext::InitContext(bool enableIndexing, bool enableSO)
{
	SetupContext(
		m_pDeferredContext.Get(),
		m_pSwapChain.Get(),
		m_pVertexShader.GetAddressOf(),
		m_pInputLayout.GetAddressOf(),
		m_pGeometryShader.GetAddressOf(),
		m_pRasertizerState.GetAddressOf(),
		m_pPixelShader.GetAddressOf(),
		m_pDepthStencilState.GetAddressOf(),
		m_pBlendState.GetAddressOf());

	UINT stride = sizeof(Vertex);
	UINT offset = 0;

	m_pDeferredContext->OMSetRenderTargets(1, m_pRenderTargetView.GetAddressOf(), nullptr);
	m_pDeferredContext->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	m_pDeferredContext->IASetVertexBuffers(0, 1, m_pIAVertexBuffer.GetAddressOf(), &stride, &offset);
	if (enableSO && !enableIndexing) {
		m_pDeferredContext->SOSetTargets(1, m_pSOVertexBuffer_A.GetAddressOf(), &offset);
	}
	else if (enableIndexing && !enableSO) {
		m_pDeferredContext->IASetIndexBuffer(m_pIBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
	}

	//Command Setup
	m_pDeferredContext->ClearRenderTargetView(m_pRenderTargetView.Get(), m_ClearColor);
	if (enableIndexing && !enableSO) {
		m_pDeferredContext->DrawIndexed(3, 0, 0);
	}
	else if (enableSO) {
		m_pDeferredContext->Draw(3, 0);
		//m_ReadSO_A = SwapIASOVertexBuffers(m_pDeferredContext.Get(), m_ReadSO_A);
		//m_pDeferredContext->ClearRenderTargetView(m_pRenderTargetView.Get(), m_ClearColor);
		//m_pDeferredContext->DrawAuto();
		//DX::ThrowIfFailed(m_pDeferredContext->FinishCommandList(FALSE, &m_pCommandList));
	}
	else {
		m_pDeferredContext->Draw(3, 0);
	}
	DX::ThrowIfFailed(m_pDeferredContext->FinishCommandList(FALSE, &m_pCommandList));
	return true;
}

bool D3D11GraphicsContext::SetupRenderingPipeline(bool enableIndexing, bool enableSO)
{
	std::vector<uint8_t> vsBytecode;
	std::vector<uint8_t> fsBytecode;
	//std::vector<uint8_t> hsBytecode;
	//std::vector<uint8_t> dsBytecode;
	std::vector<uint8_t> gsBytecode;
	SetupVertexShader(
		m_pDevice.Get(),
		"VertexShader.cso", 
		&vsBytecode, 
		m_pVertexShader.GetAddressOf());
	SetupInputAssembler(
		m_pDevice.Get(), 
		vsBytecode,
		m_pInputLayout.GetAddressOf());
	//SetupHullShader("HullShader.cso", &hsBytecode);
	//SetupTessallator();
	//SetupDomainShader("DomainShader.cso", &dsBytecode);
	CreateIAVertexBuffer();

	if (enableIndexing) {
		CreateIndexBuffer();
	}
	else if(enableSO){
		CreateSOVertexBuffers();
	}

	if(enableSO){
		SetupGeometryShaderWithStreamOutput(
			m_pDevice.Get(),
			"GeometryShader.cso",
			&gsBytecode,
			&m_pGeometryShader);
	}
	else {
		SetupGeometryShader(
			m_pDevice.Get(),
			"GeometryShader.cso", 
			&gsBytecode,
			m_pGeometryShader.GetAddressOf());
	}

	SetupRasterizer(
		m_pSwapChain.Get(),
		&m_pRasertizerState);
	SetupPixelShader(
		m_pDevice.Get(),
		"PixelShader.cso", 
		&fsBytecode,
		m_pPixelShader.GetAddressOf());
	SetupOutputMerger(
		m_pDevice.Get(), 
		m_pSwapChain.Get(),
		&m_pDepthStencil,
		&m_pBlendState);

	return true;
}

bool D3D11GraphicsContext::InitRenderingPipeline(bool enableIndexing, bool enableSO)
{
	//Init further dependencies prior to rendering here 
	//(e.g. CUDA/OpenCL/Optix Interop & D3D11<->D3D12 mergers)
	InitContext(enableIndexing, enableSO); //pass in context and manual settings
	return true;
}

bool D3D11GraphicsContext::TerminateRenderingPipeline(bool enableIndexing, bool enableSO)
{
	m_pAdapter->Release();
	m_pDevice->Release();
	m_pImmediateContext->Release();
	m_pRenderTargetView->Release();
	m_pSwapChain->Release();
	m_pIAVertexBuffer->Release();
	if (enableIndexing) {
		m_pIBuffer->Release();
	}
	if (enableSO) {
		m_pSOVertexBuffer_A->Release();
		m_pSOVertexBuffer_B->Release();
	}
	m_pVertexShader->Release();
	m_pGeometryShader->Release();
	m_pPixelShader->Release();
	m_pInputLayout->Release();
	return true;
}

void D3D11GraphicsContext::Render()
{
	m_pImmediateContext->ExecuteCommandList(m_pCommandList.Get(), TRUE);
	DX::ThrowIfFailed(m_pSwapChain->Present(1, 0));
}

bool D3D11GraphicsContext::SwapIASOVertexBuffers(ID3D11DeviceContext* context, bool readABuffer)
{
	UINT stride = sizeof(Vertex);
	UINT offset = 0;
	ID3D11Buffer* nb = NULL;
	context->SOSetTargets(1, &nb, 0);
	if (readABuffer) {
		m_pImmediateContext->IASetVertexBuffers(0, 1, m_pSOVertexBuffer_A.GetAddressOf(), &stride, &offset);
		m_pImmediateContext->SOSetTargets(1, m_pSOVertexBuffer_B.GetAddressOf(), &offset);
		readABuffer = !readABuffer;
	}
	else {
		m_pImmediateContext->IASetVertexBuffers(0, 1, m_pSOVertexBuffer_B.GetAddressOf(), &stride, &offset);
		m_pImmediateContext->SOSetTargets(1, m_pSOVertexBuffer_A.GetAddressOf(), &offset);
		readABuffer = !readABuffer;
	}
	return readABuffer;
}

std::vector<IDXGIAdapter*> D3D11GraphicsContext::EnumerateAdapters()
{
	Microsoft::WRL::ComPtr<IDXGIAdapter> pAdapter;
	std::vector<IDXGIAdapter*> vpAdapters;
	Microsoft::WRL::ComPtr<IDXGIFactory2> pFactory2;

	DX::ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&pFactory2)));
	for (UINT i = 0; 
		pFactory2->EnumAdapters(i, pAdapter.GetAddressOf()) != DXGI_ERROR_NOT_FOUND; ++i) {
		vpAdapters.push_back(pAdapter.Get());
	}

	return vpAdapters;
}

DXGI_MODE_DESC* D3D11GraphicsContext::GetAdapterDisplayMode(IDXGIAdapter* adapter, DXGI_FORMAT format)
{
	//Adapter output (typically a monitor)
	IDXGIOutput* pOutput = NULL;

	DX::ThrowIfFailed(adapter->EnumOutputs(0, &pOutput));

	UINT numModes = 0;
	DXGI_MODE_DESC* displayModes = NULL;
	//DXGI_FORMAT format = DXGI_FORMAT_R32G32B32A32_FLOAT;

	DX::ThrowIfFailed(pOutput->GetDisplayModeList(format, 0, &numModes, NULL));
	displayModes = new DXGI_MODE_DESC[numModes];
	DX::ThrowIfFailed(pOutput->GetDisplayModeList(format, 0, &numModes, displayModes));

	return displayModes;
}

LRESULT D3D11GraphicsContext::HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg) {
	case WM_SIZE:
	{
		OnResize();
	}
	break;
	case WM_PAINT:
	{
		OnPaint();
		CaptureCursor();
	}
	break;
	case WM_LBUTTONDOWN:
	{
		OnLButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
	}
	break;
	case WM_LBUTTONUP:
	{
		OnLButtonUp();
	}
	break;
	case WM_MOUSEMOVE:
	{
		OnMouseMove(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (DWORD)wParam);
	}
	break;
	case WM_SYSKEYDOWN:
	case WM_KEYDOWN:
	{
		bool alt = (::GetAsyncKeyState(VK_MENU) & 0x8000) != 0;

		switch (wParam) {
		case 'C':
		{
			if (alt) {
				m_CaptureCursor = !m_CaptureCursor;
				CaptureCursor();
			}
		}
		break;
		case VK_ESCAPE:
		{
			OnDestroy();
			DestroyWindow(m_hWnd);
		}
		}
	}
	break;
	case WM_SYSCHAR:
		break;
	case WM_DESTROY:
	{
		PostQuitMessage(0);
	}
	break;
	default:
	{
		DefWindowProcW(m_hWnd, uMsg, wParam, lParam);
	}
	break;
	}
	return 0;
}

IDXGIAdapter* D3D11GraphicsContext::GetDiscreteAdapter()
{
	std::vector<IDXGIAdapter*> vpAdapters = EnumerateAdapters();
	int adId = 0;
	size_t memCheck = 0;
	DXGI_ADAPTER_DESC ad;

	//Retrieve adapter ID with highest dedicated memory
	for (UINT i = 0; i < vpAdapters.size(); ++i) {
		vpAdapters[i]->GetDesc(&ad);
		if (ad.DedicatedSystemMemory > memCheck) {
			adId = i;
			memCheck = ad.DedicatedSystemMemory;
		}
	}
	IDXGIAdapter* adapter = vpAdapters[adId];
	return adapter;
}

bool D3D11GraphicsContext::CreateDevice(
	IDXGIAdapter** adapter,
	ID3D11Device** device,
	ID3D11DeviceContext** context)
{
	*adapter = GetDiscreteAdapter();
	
	UINT creationFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(_DEBUG)
	creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

	D3D_FEATURE_LEVEL maxSupportedFeatureLevel = D3D_FEATURE_LEVEL_9_1;
	D3D_FEATURE_LEVEL featureLevels[] = {
		D3D_FEATURE_LEVEL_11_1,
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0,
		D3D_FEATURE_LEVEL_9_3,
		D3D_FEATURE_LEVEL_9_2,
		D3D_FEATURE_LEVEL_9_1
	};

	DX::ThrowIfFailed(D3D11CreateDevice(
		*adapter,
		D3D_DRIVER_TYPE_UNKNOWN,
		nullptr,
		creationFlags,
		featureLevels,
		ARRAYSIZE(featureLevels),
		D3D11_SDK_VERSION,
		device,
		&maxSupportedFeatureLevel,
		context
	));

	return true;
}

bool D3D11GraphicsContext::CreateSwapChain(IDXGIAdapter* adapter, IDXGISwapChain1** swapChain)
{
	//Define swap chain behaviour
	DXGI_SWAP_CHAIN_DESC1 sd = {0};
	//ZeroMemory(&sd, sizeof(sd));
	sd.Width = 1280;
	sd.Height = 720;
	sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	sd.Stereo = FALSE;
	sd.SampleDesc = { 1, 0 };
	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.BufferCount = 2;
	sd.Scaling = DXGI_SCALING_STRETCH;
	sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	sd.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
	sd.Flags = 0;
	
	Microsoft::WRL::ComPtr<IDXGIOutput> pOutput;
	adapter->EnumOutputs(0, &pOutput);

	Microsoft::WRL::ComPtr<IDXGIFactory2> pFactory;
	DX::ThrowIfFailed(adapter->GetParent(IID_PPV_ARGS(&pFactory)));
	DX::ThrowIfFailed(pFactory->CreateSwapChainForHwnd(
		m_pDevice.Get(),
		m_hWnd,
		&sd,
		NULL,
		pOutput.Get(),
		swapChain
	));

	pOutput->Release();

	return true;
}

bool D3D11GraphicsContext::CreateRenderTargetView(
	ID3D11Device* device,
	IDXGISwapChain1* swapChain,
	ID3D11RenderTargetView** rtv)
{
	Microsoft::WRL::ComPtr<ID3D11Texture2D> pBackBuffer;
	DX::ThrowIfFailed(swapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer)));

	DX::ThrowIfFailed(device->CreateRenderTargetView(pBackBuffer.Get(), NULL, rtv));

	return true;
}

bool D3D11GraphicsContext::CreateDeferredContext(ID3D11Device* device, ID3D11DeviceContext** context)
{
	DX::ThrowIfFailed(device->CreateDeferredContext(0, context));
	return true;
}

void D3D11GraphicsContext::SetContextRSViewports(ID3D11DeviceContext* context, D3D11_TEXTURE2D_DESC backBufferDesc)
{
	D3D11_VIEWPORT vp = {};
	vp.Width = static_cast<float>(backBufferDesc.Width);
	vp.Height = static_cast<float>(backBufferDesc.Height);
	vp.MinDepth = D3D11_MIN_DEPTH;
	vp.MaxDepth = D3D11_MAX_DEPTH;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	context->RSSetViewports(1, &vp);
}

void D3D11GraphicsContext::SetContextRSScissorRect(ID3D11DeviceContext* context, D3D11_TEXTURE2D_DESC backBufferDesc)
{
	D3D11_RECT scRect = {};
	scRect.left = 0;
	scRect.right = backBufferDesc.Width;
	scRect.top = 0;
	scRect.bottom = backBufferDesc.Height;
	context->RSSetScissorRects(1, &scRect);
}

bool D3D11GraphicsContext::CreateIAVertexBuffer()
{
	Vertex* vb = new Vertex[3];

	vb[0].position = DirectX::XMFLOAT4(0.0f, 0.5f, 0.5f, 1.0f);
	vb[0].color = DirectX::XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f);

	vb[1].position = DirectX::XMFLOAT4(0.5f, -0.5f, 0.5f, 1.0f);
	vb[1].color = DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f);

	vb[2].position = DirectX::XMFLOAT4(-0.5f, -0.5f, 0.5f, 1.0f);
	vb[2].color = DirectX::XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f);

	D3D11_BUFFER_DESC bd = {};
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(Vertex) * 9; //Vertex Size: 32 bytes - Default Count: 3, GS Output: 9
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_STREAM_OUTPUT;
	bd.CPUAccessFlags = 0;
	bd.MiscFlags = 0;
	bd.StructureByteStride = 0;
	D3D11_SUBRESOURCE_DATA srd = {};
	srd.pSysMem = vb;
	srd.SysMemPitch = 0;
	srd.SysMemSlicePitch = 0;
	DX::ThrowIfFailed(m_pDevice->CreateBuffer(&bd, &srd, &m_pIAVertexBuffer));

	return true;
}

bool D3D11GraphicsContext::CreateSOVertexBuffers()
{
	D3D11_BUFFER_DESC bd = {};
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(Vertex) * 9; //Vertex Size: 32 bytes - Default Count: 3, GS Output: 9
	bd.BindFlags =  D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_STREAM_OUTPUT;
	bd.CPUAccessFlags = 0;
	bd.MiscFlags = 0;
	bd.StructureByteStride = 0;
	DX::ThrowIfFailed(m_pDevice->CreateBuffer(&bd, nullptr, &m_pSOVertexBuffer_A));
	DX::ThrowIfFailed(m_pDevice->CreateBuffer(&bd, nullptr, &m_pSOVertexBuffer_B));

	return true;
}

bool D3D11GraphicsContext::CreateIndexBuffer()
{
	unsigned int iBuf[] = { 0, 1, 2 };

	D3D11_BUFFER_DESC bd = {};
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(unsigned int) * 3;
	bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	bd.CPUAccessFlags = 0;
	bd.MiscFlags = 0;

	D3D11_SUBRESOURCE_DATA srd = {};
	srd.pSysMem = iBuf;
	srd.SysMemPitch = 0;
	srd.SysMemSlicePitch = 0;
	DX::ThrowIfFailed(m_pDevice->CreateBuffer(&bd, &srd, &m_pIBuffer));

	return true;
}

bool D3D11GraphicsContext::CreateConstantBuffer()
{
	VS_ConstantBuffer vsCB = {};
	vsCB.matWrapper;
	vsCB.fCountWrapper = 1.0f;
	vsCB.vecWrapper = DirectX::XMFLOAT4(1, 2, 3, 4);
	vsCB.fWrapperA = 3.0f;
	vsCB.fWrapperB = 2.0f;
	vsCB.fWrapperC = 4.0f;

	D3D11_BUFFER_DESC bd = {};
	bd.Usage = D3D11_USAGE_DYNAMIC;
	bd.ByteWidth = sizeof(VS_ConstantBuffer);
	bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	bd.MiscFlags = 0;
	bd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA srd = {};
	srd.pSysMem = &vsCB;
	srd.SysMemPitch = 0;
	srd.SysMemSlicePitch = 0;

	DX::ThrowIfFailed(m_pDevice->CreateBuffer(&bd, &srd, &m_pCBuffer));
	m_pImmediateContext->VSSetConstantBuffers(0, 1, &m_pCBuffer);

	return true;
}

bool D3D11GraphicsContext::CreateTexture()
{
	D3D11_TEXTURE2D_DESC t2dd = {};
	t2dd.Width = 256;
	t2dd.Height = 256;
	t2dd.MipLevels = t2dd.ArraySize = 1;
	t2dd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	t2dd.SampleDesc = { 1,0 };
	t2dd.Usage = D3D11_USAGE_DYNAMIC;
	t2dd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	t2dd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	t2dd.MiscFlags = 0;

	DX::ThrowIfFailed(m_pDevice->CreateTexture2D(&t2dd, NULL, &m_pTexture2D));

	return true;
}

bool D3D11GraphicsContext::SetupVertexShader(
	ID3D11Device* device,
	std::string filePath, 
	std::vector<uint8_t>* bytecode,
	ID3D11VertexShader** shader)
{
	if (GetBytecode(filePath, bytecode)) {
		DX::ThrowIfFailed(device->CreateVertexShader(
			bytecode->data(),
			bytecode->size(),
			nullptr,
			shader));

		return true;
	}
	return false;
}

bool D3D11GraphicsContext::SetupHullShader(
	ID3D11Device* device,
	std::string filePath,
	std::vector<uint8_t>* bytecode,
	ID3D11HullShader** shader)
{
	if (GetBytecode(filePath, bytecode)) {
		DX::ThrowIfFailed(device->CreateHullShader(
			bytecode->data(),
			bytecode->size(),
			nullptr,
			shader));

		return true;
	}
	return false;
}

bool D3D11GraphicsContext::SetupTessallator()
{
	return false;
}

bool D3D11GraphicsContext::SetupDomainShader(
	ID3D11Device* device,
	std::string filePath,
	std::vector<uint8_t>* bytecode,
	ID3D11DomainShader** shader)
{
	if (GetBytecode(filePath, bytecode)) {
		DX::ThrowIfFailed(device->CreateDomainShader(
			bytecode->data(),
			bytecode->size(),
			nullptr,
			shader));

		return true;
	}
	return false;
}

bool D3D11GraphicsContext::SetupGeometryShader(
	ID3D11Device* device,
	std::string filePath,
	std::vector<uint8_t>* bytecode,
	ID3D11GeometryShader** shader)
{
	if (GetBytecode(filePath, bytecode)) {
		DX::ThrowIfFailed(device->CreateGeometryShader(
			bytecode->data(),
			bytecode->size(),
			nullptr,
			shader));

		return true;
	}
	return false;
}

bool D3D11GraphicsContext::SetupGeometryShaderWithStreamOutput(
	ID3D11Device* device,
	std::string filePath,
	std::vector<uint8_t>* bytecode,
	ID3D11GeometryShader** shader)
{
	if (GetBytecode(filePath, bytecode)) {
		D3D11_SO_DECLARATION_ENTRY soEntries[2];

		soEntries[0].Stream = 0;
		soEntries[0].SemanticName = "SV_POSITION";
		soEntries[0].SemanticIndex = 0;
		soEntries[0].StartComponent = 0;
		soEntries[0].ComponentCount = 4;
		soEntries[0].OutputSlot = 0;

		soEntries[1].Stream = 0;
		soEntries[1].SemanticName = "COLOR";
		soEntries[1].SemanticIndex = 0;
		soEntries[1].StartComponent = 0;
		soEntries[1].ComponentCount = 4;
		soEntries[1].OutputSlot = 0;

		DX::ThrowIfFailed(device->CreateGeometryShaderWithStreamOutput(
			bytecode->data(),
			bytecode->size(),
			soEntries,
			ARRAYSIZE(soEntries),
			NULL,
			0,
			0,
			NULL,
			shader));
	}

	return true;
}

bool D3D11GraphicsContext::SetupRasterizer(
	IDXGISwapChain1* swapChain, 
	ID3D11RasterizerState** rasterizerState)
{
	D3D11_RASTERIZER_DESC rd = {};
	rd.FillMode = D3D11_FILL_SOLID;
	rd.CullMode = D3D11_CULL_BACK;
	rd.FrontCounterClockwise = false;
	rd.DepthBias = false;
	rd.DepthBiasClamp = 0;
	rd.SlopeScaledDepthBias = 0;
	rd.DepthClipEnable = true;
	rd.ScissorEnable = true;
	rd.MultisampleEnable = false;
	rd.AntialiasedLineEnable = false;
	DX::ThrowIfFailed(m_pDevice->CreateRasterizerState(&rd, rasterizerState));

	return true;
}

bool D3D11GraphicsContext::SetupPixelShader(
	ID3D11Device* device,
	std::string filePath, 
	std::vector<uint8_t>* bytecode,
	ID3D11PixelShader** shader)
{
	if (GetBytecode(filePath, bytecode)) {
		DX::ThrowIfFailed(device->CreatePixelShader(
			bytecode->data(),
			bytecode->size(),
			nullptr,
			shader));

		return true;
	}
	return false;
}

bool D3D11GraphicsContext::SetupOutputMerger(
	ID3D11Device* device,
	IDXGISwapChain1* swapChain,
	ID3D11Texture2D** depthStencil, 
	ID3D11BlendState** blendState)
{
	Microsoft::WRL::ComPtr<ID3D11Texture2D> pBackBuffer;
	swapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
	D3D11_TEXTURE2D_DESC scbbd = {};
	pBackBuffer->GetDesc(&scbbd);
	DXGI_FORMAT stencilFormats[2] =
	{	DXGI_FORMAT_D24_UNORM_S8_UINT,
		DXGI_FORMAT_D32_FLOAT_S8X24_UINT };
	UINT formatCheck;
	int stencilFormatId = 0;
	HRESULT hr;
	for (int i = 0; i < ARRAYSIZE(stencilFormats); ++i) {
		hr = device->CheckFormatSupport(stencilFormats[i], &formatCheck);
		if (hr == S_OK && (formatCheck & D3D11_FORMAT_SUPPORT_DEPTH_STENCIL)) {
			stencilFormatId = i;
			break;
		}
	}

	D3D11_TEXTURE2D_DESC dd = {};
	dd.Width = scbbd.Width;
	dd.Height = scbbd.Height;
	dd.MipLevels = 1;
	dd.ArraySize = 1;
	dd.Format = stencilFormats[stencilFormatId];
	dd.SampleDesc = { 1, 0 };
	dd.Usage = D3D11_USAGE_DEFAULT;
	dd.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	dd.CPUAccessFlags = 0;
	dd.MiscFlags = 0;
	DX::ThrowIfFailed(device->CreateTexture2D(&dd, NULL, depthStencil));

	D3D11_DEPTH_STENCIL_DESC dsd = {};
	dsd.DepthEnable = true;
	dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	dsd.DepthFunc = D3D11_COMPARISON_LESS;
	
	dsd.StencilEnable = true;
	dsd.StencilReadMask = 0xFF;
	dsd.StencilWriteMask = 0xFF;
	
	dsd.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	dsd.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_INCR;
	dsd.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	dsd.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;

	dsd.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	dsd.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_INCR;
	dsd.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	dsd.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	DX::ThrowIfFailed(m_pDevice->CreateDepthStencilState(&dsd, &m_pDepthStencilState));

	D3D11_BLEND_DESC bd = {};
	bd.RenderTarget[0].BlendEnable = FALSE;
	bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
	DX::ThrowIfFailed(m_pDevice->CreateBlendState(&bd, blendState));

	return true;
}

bool D3D11GraphicsContext::SetupInputAssembler(
	ID3D11Device* device,
	std::vector<uint8_t> vsBytecode,
	ID3D11InputLayout** inputLayout)
{
	D3D11_INPUT_ELEMENT_DESC layout[2];

	layout[0].SemanticName = "POSITION";
	layout[0].SemanticIndex = 0;
	layout[0].Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	layout[0].InputSlot = 0;
	layout[0].AlignedByteOffset = 0;
	layout[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	layout[0].InstanceDataStepRate = 0;

	layout[1].SemanticName = "COLOR";
	layout[1].SemanticIndex = 0;
	layout[1].Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	layout[1].InputSlot = 0;
	layout[1].AlignedByteOffset = sizeof(float) * 4;
	layout[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	layout[1].InstanceDataStepRate = 0;

	device->CreateInputLayout(
		layout, ARRAYSIZE(layout), 
		vsBytecode.data(), vsBytecode.size(),
		inputLayout);

	return true;
}
