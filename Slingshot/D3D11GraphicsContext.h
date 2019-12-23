#pragma once
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "GraphicsContext.h"

#include <d3d11.h>
#include "d3d11_1.h"
#include <dxgi1_6.h>

#include <vector>

#include "Helpers.h"
#include "GraphicsWrappers.h"
#include "CUDAContextScheduler.cuh"
#include "FileHandler.h"

class D3D11GraphicsContext : public GraphicsContext
{
public:
	D3D11GraphicsContext(HWND hwnd);
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
	void CaptureCursor();

	bool SetupD3D11();
	bool SetupContext(
		ID3D11DeviceContext* context,
		IDXGISwapChain1* swapChain,
		ID3D11VertexShader** vs,
		ID3D11InputLayout** il,
		ID3D11GeometryShader** gs,
		ID3D11RasterizerState** rs,
		ID3D11PixelShader** ps,
		ID3D11DepthStencilState** dss,
		ID3D11BlendState** bs);
	bool InitContext(bool enableIndexing = true, bool enableSO = false);
	bool SetupRenderingPipeline(bool enableIndexing = true, bool enableSO = false);
	bool InitRenderingPipeline(bool enableIndexing = true, bool enableSO = false);
	bool TerminateRenderingPipeline(bool enableIndexing = true, bool enableSO = false);
	void Render();
	bool SwapIASOVertexBuffers(ID3D11DeviceContext* context, bool readABuffer = true);

	IDXGIAdapter* GetDiscreteAdapter();
	bool CreateDevice(IDXGIAdapter** adapter, ID3D11Device** device, ID3D11DeviceContext** context);
	bool CreateSwapChain(IDXGIAdapter* adapter, IDXGISwapChain1** swapChain);
	bool CreateRenderTargetView(
		ID3D11Device* device,
		IDXGISwapChain1* swapChain, 
		ID3D11RenderTargetView** rtv);
	bool CreateDeferredContext(ID3D11Device* device, ID3D11DeviceContext** context);

	void SetContextRSViewports(ID3D11DeviceContext* context, D3D11_TEXTURE2D_DESC desc);
	void SetContextRSScissorRect(ID3D11DeviceContext* context, D3D11_TEXTURE2D_DESC desc);

	bool CreateIAVertexBuffer();
	bool CreateSOVertexBuffers();

	bool CreateIndexBuffer();
	bool CreateConstantBuffer();
	
	bool CreateTexture();

	//Rendering Pipeline:
	bool SetupInputAssembler(ID3D11Device* device,
		std::vector<uint8_t> vsBytecode,
		ID3D11InputLayout** inputLayout);
	bool SetupVertexShader(
		ID3D11Device* device,
		std::string filePath, 
		std::vector<uint8_t>* bytecode,
		ID3D11VertexShader** shader);
	bool SetupHullShader(
		ID3D11Device* device,
		std::string filePath, 
		std::vector<uint8_t>* bytecode,
		ID3D11HullShader** shader);
	bool SetupTessallator();
	bool SetupDomainShader(
		ID3D11Device* device,
		std::string filePath,
		std::vector<uint8_t>* bytecode,
		ID3D11DomainShader** shader);
	bool SetupGeometryShader(
		ID3D11Device* device,
		std::string filePath,
		std::vector<uint8_t>* bytecode,
		ID3D11GeometryShader** shader);
	bool SetupGeometryShaderWithStreamOutput(
		ID3D11Device* device,
		std::string filePath,
		std::vector<uint8_t>* bytecode,
		ID3D11GeometryShader** shader);
	bool SetupRasterizer(
		IDXGISwapChain1* swapChain, 
		ID3D11RasterizerState** rasterizerState);
	bool SetupPixelShader(
		ID3D11Device* device,
		std::string filePath,
		std::vector<uint8_t>* bytecode,
		ID3D11PixelShader** shader);
	bool SetupOutputMerger(
		ID3D11Device* device,
		IDXGISwapChain1* swapChain,
		ID3D11Texture2D** depthStencil,
		ID3D11BlendState** blendState);

	//Window behaviour:
	bool m_CaptureCursor = false;

	//Pipeline Managed behaviour:
	//--------------------------
	//Input Assembly Stage:
	bool m_EnableIndexing = false;
	//Stream Output Stage:
	bool m_EnableSO = true;	
	bool m_SwapIASOBuffers = false;

	//Pipeline Auto behaviour:
	//--------------------------
	//IA<->SO Vertex Buffer Handling
	bool m_ReadSO_A = true;

	const float m_ClearColor[4] = { 0.3f, 0.5f, 0.92f, 1.0f };

	Microsoft::WRL::ComPtr<ID3D11Device> m_pDevice;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_pImmediateContext;
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_pRenderTargetView;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_pDeferredContext;
	Microsoft::WRL::ComPtr<ID3D11CommandList> m_pCommandList;

	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pIAVertexBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pSOVertexBuffer_A;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pSOVertexBuffer_B;

	Microsoft::WRL::ComPtr<IDXGIAdapter> m_pAdapter;
	Microsoft::WRL::ComPtr<IDXGISwapChain1> m_pSwapChain;

	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pIBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_pCBuffer;

	//----------------Rendering Pipeline----------------
	//Input Assembler
	Microsoft::WRL::ComPtr<ID3D11InputLayout> m_pInputLayout;
	//Vertex Shader
	Microsoft::WRL::ComPtr<ID3D11VertexShader> m_pVertexShader;
	//Hull Shader
	Microsoft::WRL::ComPtr<ID3D11HullShader> m_pHullShader;
	//Domain Shader
	Microsoft::WRL::ComPtr<ID3D11DomainShader> m_pDomainShader;
	//Geometry Shader
	Microsoft::WRL::ComPtr<ID3D11GeometryShader> m_pGeometryShader;
	//Raserizer
	Microsoft::WRL::ComPtr<ID3D11RasterizerState> m_pRasertizerState;
	//Pixel Shader
	Microsoft::WRL::ComPtr<ID3D11PixelShader> m_pPixelShader;
	//Output Merger
	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_pDepthStencil;
	Microsoft::WRL::ComPtr<ID3D11BlendState> m_pBlendState;
	Microsoft::WRL::ComPtr<ID3D11DepthStencilState> m_pDepthStencilState;
	//--------------------------------------------------

	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_pTexture2D;
	Microsoft::WRL::ComPtr<ID3D11Resource> m_pTexture;
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_pTextureView;

	std::unique_ptr<HC::D3D11DeviceInteropContext> m_pDeviceInteropContext;
private:
	HWND m_hWnd;
};

