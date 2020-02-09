#pragma once
#include "GraphicsContext.h"
#include "Helpers.h"

#include <d3d11.h>
#include "d3d11_1.h"
#include <dxgi1_6.h>

class D3D11Context
{
public:
	static D3D11Context * Create(HWND hWnd);
	bool Initialize();
	void StartFrameRender();
	void EndFrameRender();
	void Shutdown();
private:
	D3D11Context(HWND hWnd);

	std::vector<IDXGIAdapter*> QueryAdapters();
	IDXGIAdapter* GetLatestDiscreteAdapter();

	void CreateDeviceAndContext(
		IDXGIAdapter* adapter,
		ID3D11Device** device,
		ID3D11DeviceContext** immediateContext);

	void CreateSwapChain(
		IDXGIAdapter* adapter,
		IDXGISwapChain1** swapChain,
		HWND hWnd);

	void CreateRenderTargetView(
		ID3D11Device* device,
		IDXGISwapChain1* swapChain,
		ID3D11RenderTargetView** rtv);

	float m_clearColor[4];
	Microsoft::WRL::ComPtr<ID3D11Device> m_pDevice;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_pImmediateContext;
	Microsoft::WRL::ComPtr<IDXGISwapChain1> m_pSwapChain;
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_pRenderTargetView;
};
