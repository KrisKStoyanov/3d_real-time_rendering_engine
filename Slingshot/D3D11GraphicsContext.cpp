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
	CreateRenderTargetView();
	
	//Setup rendering pipeline:
	CreateVertexShader("VertexShader.cso");
	CreateFragmentShader("PixelShader.cso");
	
	CreateVertexBuffer();

	SetupInputAssembler();

	//Setup interop pipeline extension
	//m_pDeviceInteropContext = new HC::D3D11DeviceInteropContext();

	//HC::InvokeRenderKernel(m_pDeviceInteropContext->m_ScreenBuffer, 1280, 720);
}

void D3D11GraphicsContext::OnDestroy()
{
	m_pAdapter->Release();
	m_pDevice->Release();
	m_pDeviceContext->Release();
	m_pSwapChain->Release();
	m_pRenderTargetView->Release();
}

void D3D11GraphicsContext::OnPaint()
{
	m_pDeviceContext->OMSetRenderTargets(1, m_pRenderTargetView.GetAddressOf(), nullptr);
	m_pDeviceContext->ClearRenderTargetView(m_pRenderTargetView.Get(), m_ClearColor);

	m_pDeviceContext->Draw(3, 0);
	DX::ThrowIfFailed(m_pSwapChain->Present(1, 0));
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
	Microsoft::WRL::ComPtr<IDXGIFactory2> pFactory2;

	DX::ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&pFactory2)));
	for (UINT i = 0; 
		pFactory2->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
		vpAdapters.push_back(pAdapter);
	}

	DX::SafeRelease(&pAdapter);
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
		return 0;
	}
	break;
	case WM_SYSCHAR:
		break;
	case WM_DESTROY:
	{
		PostQuitMessage(0);
		return 0;
	}
	break;
	default:
	{
		DefWindowProcW(m_hWnd, uMsg, wParam, lParam);
	}
	break;
	}
}

void D3D11GraphicsContext::CreateDevice()
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
	m_pAdapter = vpAdapters[adId];
	
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
		m_pAdapter.Get(),
		D3D_DRIVER_TYPE_UNKNOWN,
		nullptr,
		creationFlags,
		featureLevels,
		ARRAYSIZE(featureLevels),
		D3D11_SDK_VERSION,
		&m_pDevice,
		&maxSupportedFeatureLevel,
		&m_pDeviceContext
	));
}

void D3D11GraphicsContext::CreateSwapChain()
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
	m_pAdapter->EnumOutputs(0, &pOutput);

	Microsoft::WRL::ComPtr<IDXGIFactory2> pFactory2;
	DX::ThrowIfFailed(m_pAdapter->GetParent(IID_PPV_ARGS(&pFactory2)));
	DX::ThrowIfFailed(pFactory2->CreateSwapChainForHwnd(
		m_pDevice.Get(),
		m_hWnd,
		&sd,
		NULL,
		pOutput.Get(),
		&m_pSwapChain
	));

	pOutput->Release();
}

void D3D11GraphicsContext::CreateRenderTargetView()
{
	Microsoft::WRL::ComPtr<ID3D11Texture2D> pBackBuffer;
	DX::ThrowIfFailed(m_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer)));

	DX::ThrowIfFailed(m_pDevice->CreateRenderTargetView(pBackBuffer.Get(), NULL, &m_pRenderTargetView));

	D3D11_TEXTURE2D_DESC backBufferDesc = { 0 };
	pBackBuffer->GetDesc(&backBufferDesc);

	D3D11_VIEWPORT vp;
	vp.Width = static_cast<float>(backBufferDesc.Width);
	vp.Height = static_cast<float>(backBufferDesc.Height);
	vp.MinDepth = D3D11_MIN_DEPTH;
	vp.MaxDepth = D3D11_MAX_DEPTH;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	m_pDeviceContext->RSSetViewports(1, &vp);
}

void D3D11GraphicsContext::CreateVertexBuffer()
{
	Vertex* vBuf = new Vertex[3];

	vBuf[0].position = DirectX::XMFLOAT4(0.0f, 0.5f, 0.5f, 1.0f);
	//vBuf[0].color = DirectX::XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f);

	vBuf[1].position = DirectX::XMFLOAT4(0.5f, -0.5f, 0.5f, 1.0f);
	//vBuf[1].color = DirectX::XMFLOAT4(1.0f, 1.0f, 0.0f, 1.0f);

	vBuf[2].position = DirectX::XMFLOAT4(-0.5f, -0.5f, 0.5f, 1.0f);
	//vBuf[2].color = DirectX::XMFLOAT4(1.0f, 0.0f, 1.0f, 1.0f);

	D3D11_BUFFER_DESC bd = {};
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(Vertex) * 3;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;
	bd.MiscFlags = 0;

	D3D11_SUBRESOURCE_DATA srd = {};
	srd.pSysMem = vBuf;
	srd.SysMemPitch = 0;
	srd.SysMemSlicePitch = 0;

	DX::ThrowIfFailed(m_pDevice->CreateBuffer(&bd, &srd, &m_pVBuffer));
	UINT stride = sizeof(Vertex);
	UINT offset = 0;
	m_pDeviceContext->IASetVertexBuffers(0, 1, m_pVBuffer.GetAddressOf(), &stride, &offset);
}

void D3D11GraphicsContext::CreateIndexBuffer()
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
	m_pDeviceContext->IASetIndexBuffer(m_pIBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
}

void D3D11GraphicsContext::CreateConstantBuffer()
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
	m_pDeviceContext->VSSetConstantBuffers(0, 1, &m_pCBuffer);
}

void D3D11GraphicsContext::CreateTexture()
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
}

void D3D11GraphicsContext::CreateVertexShader(std::string filePath)
{
	vShaderBytecode = ReadFile(filePath);

	DX::ThrowIfFailed(m_pDevice->CreateVertexShader(
		vShaderBytecode.data(),
		vShaderBytecode.size(),
		nullptr,
		&m_pVertexShader));

	m_pDeviceContext->VSSetShader(m_pVertexShader.Get(), nullptr, 0);
}

void D3D11GraphicsContext::CreateFragmentShader(std::string filePath)
{
	std::vector<uint8_t> shaderBytecode = ReadFile(filePath);

	DX::ThrowIfFailed(m_pDevice->CreatePixelShader(
		shaderBytecode.data(),
		shaderBytecode.size(),
		nullptr,
		&m_pPixelShader));

	m_pDeviceContext->PSSetShader(m_pPixelShader.Get(), nullptr, 0);
}

void D3D11GraphicsContext::SetupInputAssembler()
{
	D3D11_INPUT_ELEMENT_DESC layout[1];

	layout[0].SemanticName = "POSITION";
	layout[0].SemanticIndex = 0;
	layout[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	layout[0].InputSlot = 0;
	layout[0].AlignedByteOffset = 0;
	layout[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	layout[0].InstanceDataStepRate = 0;

	m_pDevice->CreateInputLayout(
		layout, ARRAYSIZE(layout), 
		vShaderBytecode.data(), vShaderBytecode.size(),
		&m_pInputLayout);
	m_pDeviceContext->IASetInputLayout(m_pInputLayout.Get());

	m_pDeviceContext->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
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
