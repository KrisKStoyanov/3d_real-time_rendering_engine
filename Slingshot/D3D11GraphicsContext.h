#pragma once
#include "GraphicsContext.h"

#include <d3d11.h>
#include <dxgi1_6.h>
#include <vector>

#include "Helpers.h"

class D3D11GraphicsContext : public GraphicsContext
{
public:
	D3D11GraphicsContext();
	~D3D11GraphicsContext();

	std::vector<IDXGIAdapter*> EnumerateAdapters();
	DXGI_MODE_DESC* GetAdapterDisplayMode(IDXGIAdapter* adapter, DXGI_FORMAT format);
	
	void OnCreate(HWND hwnd);
	void OnDestroy();
	void OnPaint();
	void OnResize();
	void OnLButtonDown(int pixelX, int pixelY, DWORD flags);
	void OnLButtonUp();
	void OnMouseMove(int pixelX, int pixelY, DWORD flags);

	void CreateDevice();
	void CreateSwapChain();

	HWND m_hWnd;

	Microsoft::WRL::ComPtr<ID3D11Device> m_pDevice;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_pDeviceContext;

	Microsoft::WRL::ComPtr<IDXGIAdapter> m_pAdapter;
	Microsoft::WRL::ComPtr<IDXGIOutput> m_pOutput;
	Microsoft::WRL::ComPtr<IDXGISwapChain1> m_pSwapChain;
private:
};

