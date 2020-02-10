#pragma once
#include "GraphicsContext.h"
#include "Helpers.h"
#include "Macros.h"

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

	ID3D11Device* GetDevice();
	ID3D11DeviceContext* GetDeviceContext();
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
		HWND hWnd, UINT winWidth, UINT winHeight);

	void CreateRenderTargetView(
		ID3D11Device* device,
		IDXGISwapChain1* swapChain,
		ID3D11RenderTargetView** rtv);

	void CreateDepthStencilBuffer(
		ID3D11Device* device,
		ID3D11Texture2D** depthStencilBuffer,
		UINT winWidth, UINT winHeight);

	void CreateDepthStencilView(
		ID3D11Device* device,
		ID3D11Texture2D* depthStencilBuffer,
		ID3D11DepthStencilView** depthStencilView);

	void SetupViewport(
		D3D11_VIEWPORT viewport, 
		UINT winWidth, UINT winHeight);

	float m_clearColor[4];
	Microsoft::WRL::ComPtr<ID3D11Device> m_pDevice;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_pImmediateContext;
	Microsoft::WRL::ComPtr<IDXGISwapChain1> m_pSwapChain;
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_pRenderTargetView;

	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_pDepthStencilBuffer;
	Microsoft::WRL::ComPtr<ID3D11DepthStencilView> m_pDepthStencilView;
	D3D11_VIEWPORT m_viewport;
};
