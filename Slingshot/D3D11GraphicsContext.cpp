#include "D3D11GraphicsContext.h"

D3D11GraphicsContext::D3D11GraphicsContext()
{
}

D3D11GraphicsContext::~D3D11GraphicsContext()
{
}

void D3D11GraphicsContext::OnCreate(HWND hwnd)
{
	m_hWnd = hwnd;
	CreateDevice();
	CreateSwapChain();
}

void D3D11GraphicsContext::OnDestroy()
{
}

void D3D11GraphicsContext::OnPaint()
{
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

std::vector<IDXGIAdapter*> D3D11GraphicsContext::EnumerateAdapters()
{
	IDXGIAdapter* pAdapter;
	std::vector<IDXGIAdapter*> vpAdapters;
	IDXGIFactory1* pFactory = NULL;

	DX::ThrowIfFailed(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory));
	for (UINT i = 0; 
		pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
		vpAdapters.push_back(pAdapter);
	}

	DX::SafeRelease(&pFactory);	
	return vpAdapters;
}

DXGI_MODE_DESC* D3D11GraphicsContext::GetAdapterDisplayMode(IDXGIAdapter* adapter, DXGI_FORMAT format)
{
	//Adapter output (typically a monitor)
	IDXGIOutput* pOutput = NULL;
	HRESULT hr;

	DX::ThrowIfFailed(adapter->EnumOutputs(0, &pOutput));

	UINT numModes = 0;
	DXGI_MODE_DESC* displayModes = NULL;
	//DXGI_FORMAT format = DXGI_FORMAT_R32G32B32A32_FLOAT;

	DX::ThrowIfFailed(pOutput->GetDisplayModeList(format, 0, &numModes, NULL));
	displayModes = new DXGI_MODE_DESC[numModes];
	DX::ThrowIfFailed(pOutput->GetDisplayModeList(format, 0, &numModes, displayModes));

	return displayModes;
}

void D3D11GraphicsContext::CreateDevice()
{
	UINT creationFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(_DEBUG)
	creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

	D3D_FEATURE_LEVEL featureLevels[] = {
		D3D_FEATURE_LEVEL_11_1,
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0,
		D3D_FEATURE_LEVEL_9_3,
		D3D_FEATURE_LEVEL_9_1
	};

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
	m_pAdapter = vpAdapters[adId];
	
	DX::ThrowIfFailed(D3D11CreateDevice(
		m_pAdapter.Get(),
		D3D_DRIVER_TYPE_HARDWARE,
		nullptr,
		creationFlags,
		featureLevels,
		ARRAYSIZE(featureLevels),
		D3D11_SDK_VERSION,
		&m_pDevice,
		nullptr,
		&m_pDeviceContext
	));
}

void D3D11GraphicsContext::CreateSwapChain()
{
	//Define swap chain behaviour
	DXGI_SWAP_CHAIN_DESC1 sd = {};
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
	sd.AlphaMode = DXGI_ALPHA_MODE_STRAIGHT;
	sd.Flags = 0;
		
	m_pAdapter->EnumOutputs(0, &m_pOutput);

	Microsoft::WRL::ComPtr<IDXGIFactory2> pFactory2;
	DX::ThrowIfFailed(m_pAdapter->GetParent(IID_PPV_ARGS(&pFactory2)));
	DX::ThrowIfFailed(pFactory2->CreateSwapChainForHwnd(
		m_pDevice.Get(),
		m_hWnd,
		&sd,
		NULL,
		m_pOutput.Get(),
		&m_pSwapChain
	));
}
