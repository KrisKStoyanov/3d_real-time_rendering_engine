#include "D3D11Context.h"

D3D11Context* D3D11Context::Create(HWND hWnd)
{
	return new D3D11Context(hWnd);
}

D3D11Context::D3D11Context(HWND hWnd)
{
	RECT winRect;
	GetWindowRect(hWnd, &winRect);
	UINT winWidth = winRect.right - winRect.left;
	UINT winHeight = winRect.bottom - winRect.top;

	m_clearColor[0] = 0.2f;
	m_clearColor[1] = 0.8f;
	m_clearColor[2] = 1.0f;
	m_clearColor[3] = 1.0f;

	IDXGIAdapter* pAdapter = GetLatestDiscreteAdapter();
	
	CreateDeviceAndContext(
		pAdapter, 
		m_pDevice.GetAddressOf(), 
		m_pImmediateContext.GetAddressOf());

	CreateSwapChain(
		pAdapter,
		m_pSwapChain.GetAddressOf(), 
		hWnd, winWidth, winHeight);

	CreateRenderTargetView(
		m_pDevice.Get(),
		m_pSwapChain.Get(),
		m_pRenderTargetView.GetAddressOf());

	CreateDepthStencilBuffer(
		m_pDevice.Get(),
		m_pDepthStencilBuffer.GetAddressOf(), 
		winWidth, 
		winHeight);

	CreateDepthStencilView(
		m_pDevice.Get(),
		m_pDepthStencilBuffer.Get(),
		m_pDepthStencilView.GetAddressOf());

	SetupViewport(winWidth, winHeight);

	pAdapter->Release();
}

std::vector<IDXGIAdapter*> D3D11Context::QueryAdapters()
{
	IDXGIAdapter* pAdapter;
	std::vector<IDXGIAdapter*> vpAdapters;
	IDXGIFactory2* pFactory;

	DX::ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&pFactory)));
	for (UINT i = 0;
		pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
		vpAdapters.push_back(pAdapter);
	}

	pFactory->Release();
	return vpAdapters;
}

IDXGIAdapter* D3D11Context::GetLatestDiscreteAdapter()
{
	std::vector<IDXGIAdapter*> vpAdapters = QueryAdapters();
	int adId = 0;
	size_t memCheck = 0;
	DXGI_ADAPTER_DESC ad;

	//Retrieve adapter ID with highest dedicated memory
	for (UINT i = 0; i < vpAdapters.size(); ++i) {
		vpAdapters[i]->GetDesc(&ad);
		if (ad.DedicatedVideoMemory > memCheck) {
			adId = i;
			memCheck = ad.DedicatedVideoMemory;
		}
	}
	IDXGIAdapter* adapter = vpAdapters[adId];
	return adapter;
}

void D3D11Context::CreateDeviceAndContext(
	IDXGIAdapter* adapter,
	ID3D11Device** device, 
	ID3D11DeviceContext** immediateContext)
{
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
		adapter,
		D3D_DRIVER_TYPE_UNKNOWN,
		nullptr,
		creationFlags,
		featureLevels,
		ARRAYSIZE(featureLevels),
		D3D11_SDK_VERSION,
		device,
		&maxSupportedFeatureLevel,
		immediateContext));
}

void D3D11Context::CreateSwapChain(
	IDXGIAdapter* adapter, 
	IDXGISwapChain1** swapChain, 
	HWND hWnd, UINT winWidth, UINT winHeight)
{
	DXGI_SWAP_CHAIN_DESC1 sd;
	ZeroMemory(&sd, sizeof(sd));
	sd.Width = winWidth;
	sd.Height = winHeight;
	sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	sd.Stereo = FALSE;
	sd.SampleDesc = { 1, 0 };
	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.BufferCount = 2;
	sd.Scaling = DXGI_SCALING_STRETCH;
	sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	sd.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
	sd.Flags = 0;

	IDXGIOutput* pOutput;
	adapter->EnumOutputs(0, &pOutput);

	IDXGIFactory2* pFactory;
	DX::ThrowIfFailed(adapter->GetParent(IID_PPV_ARGS(&pFactory)));
	DX::ThrowIfFailed(pFactory->CreateSwapChainForHwnd(
		m_pDevice.Get(),
		hWnd,
		&sd,
		NULL,
		pOutput,
		swapChain
	));

	pOutput->Release();
	pFactory->Release();
}

void D3D11Context::CreateRenderTargetView(
	ID3D11Device* device,
	IDXGISwapChain1* swapChain, 
	ID3D11RenderTargetView** rtv)
{
	ID3D11Texture2D* pBackBuffer;
	DX::ThrowIfFailed(swapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer)));
	if (pBackBuffer != nullptr) {
		DX::ThrowIfFailed(device->CreateRenderTargetView(pBackBuffer, nullptr, rtv));
		pBackBuffer->Release();
	}
}

void D3D11Context::CreateDepthStencilBuffer(
	ID3D11Device* device,
	ID3D11Texture2D** depthStencilBuffer,
	UINT winWidth, UINT winHeight)
{
	D3D11_TEXTURE2D_DESC depthBufferDesc;
	ZeroMemory(&depthBufferDesc, sizeof(depthBufferDesc));
	depthBufferDesc.Width = winWidth;
	depthBufferDesc.Height = winHeight;
	depthBufferDesc.MipLevels = 1;
	depthBufferDesc.ArraySize = 1;
	depthBufferDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depthBufferDesc.SampleDesc.Count = 1;
	depthBufferDesc.SampleDesc.Quality = 0;
	depthBufferDesc.Usage = D3D11_USAGE_DEFAULT;
	depthBufferDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	depthBufferDesc.CPUAccessFlags = 0;
	depthBufferDesc.MiscFlags = 0;

	DX::ThrowIfFailed(device->CreateTexture2D(&depthBufferDesc, nullptr, depthStencilBuffer));
}

void D3D11Context::CreateDepthStencilView(
	ID3D11Device* device,
	ID3D11Texture2D* depthStencilBuffer, 
	ID3D11DepthStencilView** depthStencilView)
{
	D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
	ZeroMemory(&depthStencilViewDesc, sizeof(depthStencilViewDesc));
	depthStencilViewDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	depthStencilViewDesc.Texture2D.MipSlice = 0;

	DX::ThrowIfFailed(
		device->CreateDepthStencilView(
			depthStencilBuffer, 
			&depthStencilViewDesc, 
			depthStencilView));
}

void D3D11Context::SetupViewport(UINT winWidth, UINT winHeight)
{
	D3D11_VIEWPORT viewport;
	ZeroMemory(&viewport, sizeof(viewport));
	viewport.Width = static_cast<float>(winWidth);
	viewport.Height = static_cast<float>(winHeight);
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;
	viewport.TopLeftX = 0.0f;
	viewport.TopLeftY = 0.0f;
	m_pImmediateContext->RSSetViewports(1, &viewport);
}

bool D3D11Context::Initialize()
{
	m_pImmediateContext->OMSetRenderTargets(1, 
		m_pRenderTargetView.GetAddressOf(), 
		m_pDepthStencilView.Get());

	InitializeNvAPI();
	if (m_gfxCaps.bVariablePixelRateShadingSupported) 
	{
		SetVRS(true);
	}

	return true;
}

void D3D11Context::StartFrameRender()
{
	m_pImmediateContext->ClearRenderTargetView(m_pRenderTargetView.Get(), m_clearColor);
	m_pImmediateContext->ClearDepthStencilView(m_pDepthStencilView.Get(), D3D11_CLEAR_DEPTH, 1.0f, 1);
	m_pImmediateContext->OMSetRenderTargets(1,
		m_pRenderTargetView.GetAddressOf(),
		m_pDepthStencilView.Get());
}

void D3D11Context::EndFrameRender()
{
	//V-Sync enabled
	DX::ThrowIfFailed(m_pSwapChain->Present(1, 0));
}

void D3D11Context::Shutdown()
{
	if (m_enableNvAPI)
	{
		SetVRS(false);
		ShutdownNvAPI();
	}
	m_pDepthStencilBuffer->Release();
	m_pDepthStencilView->Release();
	m_pRenderTargetView->Release();
	m_pSwapChain->Release();
	m_pDevice->Release();
	m_pImmediateContext->Release();
}

void D3D11Context::InitializeNvAPI()
{
	NvAPI_Status NvStatus = NvAPI_Initialize();
	if (NvStatus == NVAPI_OK) 
	{
		m_gfxCaps = QueryGraphicsCapabilities();
		m_enableNvAPI = true;
	}
}

void D3D11Context::ShutdownNvAPI()
{
}

NV_D3D1x_GRAPHICS_CAPS D3D11Context::QueryGraphicsCapabilities()
{
	NV_D3D1x_GRAPHICS_CAPS caps;
	memset(&caps, 0, sizeof(NV_D3D1x_GRAPHICS_CAPS));

	NvAPI_Status NvStatus = NvAPI_D3D1x_GetGraphicsCapabilities(m_pDevice.Get(), NV_D3D1x_GRAPHICS_CAPS_VER, &caps);
	if (NvStatus == NVAPI_OK)
	{
		return caps;
	}
	return caps; //print some kinda error (make macro)
}

void D3D11Context::SetVRS(bool enable)
{
	NV_D3D11_VIEWPORTS_SHADING_RATE_DESC sShadingRateDesc;
	ZeroMemory(&sShadingRateDesc, sizeof(NV_D3D11_VIEWPORTS_SHADING_RATE_DESC));
	if (enable) 
	{
		sShadingRateDesc.numViewports = 1;
		sShadingRateDesc.pViewports =
			(NV_D3D11_VIEWPORT_SHADING_RATE_DESC*)malloc(sizeof(NV_D3D11_VIEWPORT_SHADING_RATE_DESC));
		if (sShadingRateDesc.pViewports)
		{
			sShadingRateDesc.version = NV_D3D11_VIEWPORTS_SHADING_RATE_DESC_VER;
			sShadingRateDesc.pViewports[0].enableVariablePixelShadingRate = true;
			for (unsigned int i = 0; i < NV_MAX_PIXEL_SHADING_RATES; i++)
			{
				sShadingRateDesc.pViewports[0].shadingRateTable[i] = NV_PIXEL_SHADING_RATE::NV_PIXEL_X1_PER_RASTER_PIXEL;
			}
			NvAPI_Status NvStatus = NvAPI_D3D11_RSSetViewportsPixelShadingRates(m_pImmediateContext.Get(), &sShadingRateDesc);
		}
	}
	else 
	{
		sShadingRateDesc.numViewports = 0;
		NvAPI_Status NvStatus = NvAPI_D3D11_RSSetViewportsPixelShadingRates(m_pImmediateContext.Get(), &sShadingRateDesc);
	}
	m_enableVRS = enable;
}

bool D3D11Context::GetVRS()
{
	return m_enableVRS;
}

ID3D11Device* D3D11Context::GetDevice()
{
	return m_pDevice.Get();
}

ID3D11DeviceContext* D3D11Context::GetDeviceContext()
{
	return m_pImmediateContext.Get();
}
