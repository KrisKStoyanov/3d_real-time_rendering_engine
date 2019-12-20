#pragma once
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "GraphicsContext.h"

#include <d3d11.h>
#include <dxgi1_6.h>
#include <vector>

#include "Helpers.h"
#include "GraphicsWrappers.h"
#include "CUDAContextScheduler.cuh"
#include "FileHandler.h"

class D3D11GraphicsContext : public GraphicsContext
{
public:
	D3D11GraphicsContext();
	~D3D11GraphicsContext();

	std::vector<IDXGIAdapter*> EnumerateAdapters();
	DXGI_MODE_DESC* GetAdapterDisplayMode(IDXGIAdapter* adapter, DXGI_FORMAT format);
	
	virtual LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);

	virtual void OnCreate(HWND hwnd);
	virtual void OnDestroy();
	virtual void OnPaint();
	virtual void OnResize();
	virtual void OnLButtonDown(int pixelX, int pixelY, DWORD flags);
	virtual void OnLButtonUp();
	virtual void OnMouseMove(int pixelX, int pixelY, DWORD flags);

	void CreateDevice();
	void CreateSwapChain();
	void CreateRenderTargetView();

	void CreateVertexBuffer();
	void CreateIndexBuffer();
	void CreateConstantBuffer();
	
	void CreateTexture();

	void CreateVertexShader(std::string filePath);
	void CreateFragmentShader(std::string filePath);

	//Rendering Pipeline Setup:
	void SetupInputAssembler();

	void CaptureCursor();

	HWND m_hWnd;
	bool m_CaptureCursor = false;
	const float m_ClearColor[4] = { 1.0f, 0.5f, 0.32f, 1.0f };
	std::vector<uint8_t> vShaderBytecode;

	Microsoft::WRL::ComPtr<ID3D11Device> m_pDevice;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_pDeviceContext;
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_pRenderTargetView;

	Microsoft::WRL::ComPtr<IDXGIAdapter> m_pAdapter;
	Microsoft::WRL::ComPtr<IDXGISwapChain1> m_pSwapChain;

	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pVBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pIBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pCBuffer;

	Microsoft::WRL::ComPtr<ID3D11VertexShader> m_pVertexShader;
	Microsoft::WRL::ComPtr<ID3D11PixelShader> m_pPixelShader;
	Microsoft::WRL::ComPtr<ID3D11InputLayout> m_pInputLayout;

	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_pTexture2D;
	Microsoft::WRL::ComPtr<ID3D11Resource> m_pTexture;
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_pTextureView;

	HC::D3D11DeviceInteropContext* m_pDeviceInteropContext;

private:
};

