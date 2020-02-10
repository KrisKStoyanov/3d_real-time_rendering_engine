#include "D3D11Context.h"

D3D11Context* D3D11Context::Create(HWND hWnd)
{
	return new D3D11Context(hWnd);
}

D3D11Context::D3D11Context(HWND hWnd)
{
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
		hWnd);

	CreateRenderTargetView(
		m_pDevice.Get(),
		m_pSwapChain.Get(),
		m_pRenderTargetView.GetAddressOf());

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
	HWND hWnd)
{
	RECT winRect;
	GetWindowRect(hWnd, &winRect);
	UINT winWidth = winRect.right - winRect.left;
	UINT winHeight = winRect.bottom - winRect.top;

	DXGI_SWAP_CHAIN_DESC1 sd = {};
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
		DX::ThrowIfFailed(device->CreateRenderTargetView(pBackBuffer, NULL, rtv));
		pBackBuffer->Release();
	}
}

bool D3D11Context::Initialize()
{
	return true;
}

void D3D11Context::StartFrameRender()
{
	m_pImmediateContext->ClearRenderTargetView(m_pRenderTargetView.Get(), m_clearColor);
}

void D3D11Context::EndFrameRender()
{
	//V-Sync enabled
	DX::ThrowIfFailed(m_pSwapChain->Present(1, 0));
}

void D3D11Context::Shutdown()
{
	m_pRenderTargetView->Release();
	m_pSwapChain->Release();
	m_pDevice->Release();
	m_pImmediateContext->Release();
}
