#pragma once
#include "GraphicsContext.h"
#include "Helpers.h"
#include "Macros.h"

#include <d3d11.h>
#include "d3d11_1.h"
#include <dxgi1_6.h>

#include "nvapi.h"

class D3D11Context
{
public:
	static D3D11Context * Create(HWND hWnd);
	bool Initialize();
	void StartFrameRender();
	void EndFrameRender();
	void Shutdown();

	void SetVRS(bool enable);
	bool GetVRS();

	ID3D11Device* GetDevice();
	ID3D11DeviceContext* GetDeviceContext();
private:
	D3D11Context(HWND hWnd);

	std::vector<IDXGIAdapter*> QueryAdapters();
	IDXGIAdapter* GetLatestDiscreteAdapter();

	void CreateDeviceAndContext();

	void CreateSwapChain(
		HWND hWnd, UINT winWidth, UINT winHeight);

	void CreateRenderTargetView();

	void CreateDepthStencilBuffer(
		UINT winWidth, UINT winHeight);

	void CreateDepthStencilView();

	void CreateRasterizerState();
	void SetupViewport(UINT winWidth, UINT winHeight);

	float m_clearColor[4];
	Microsoft::WRL::ComPtr<ID3D11Device> m_pDevice;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_pImmediateContext;
	Microsoft::WRL::ComPtr<IDXGISwapChain1> m_pSwapChain;
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_pRenderTargetView;

	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_pDepthStencilBuffer;
	Microsoft::WRL::ComPtr<ID3D11DepthStencilView> m_pDepthStencilView;

	Microsoft::WRL::ComPtr<ID3D11RasterizerState> m_pRasterizerState;
	D3D11_VIEWPORT m_viewport;

	void InitializeNvAPI();
	void ShutdownNvAPI();
	NV_D3D1x_GRAPHICS_CAPS QueryGraphicsCapabilities();

	NV_D3D1x_GRAPHICS_CAPS m_gfxCaps;
	bool m_enableNvAPI;
	bool m_enableVRS;
};
