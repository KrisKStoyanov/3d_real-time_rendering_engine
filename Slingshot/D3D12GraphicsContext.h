#pragma once
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "GraphicsContext.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>

#include "d3dx12.h"

#include "Helpers.h"

class D3D12GraphicsContext : public GraphicsContext
{
public:
	D3D12GraphicsContext();
	~D3D12GraphicsContext();

	virtual LRESULT HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);

	virtual void OnCreate(HWND hwnd);
	virtual void OnDestroy();
	virtual void OnPaint();
	virtual void OnResize(uint32_t width, uint32_t height);
	virtual void OnLButtonDown(int pixelX, int pixelY, DWORD flags);
	virtual void OnLButtonUp();
	virtual void OnMouseMove(int pixelX, int pixelY, DWORD flags);

	void ParseCommandLineArguments();
	void EnableDebugLayer();

	Microsoft::WRL::ComPtr<IDXGIAdapter4> GetAdapter(bool useWarp);

	Microsoft::WRL::ComPtr<ID3D12Device2> CreateDevice(
		Microsoft::WRL::ComPtr<IDXGIAdapter4> adapter);

	Microsoft::WRL::ComPtr<ID3D12CommandQueue> CreateCommandQueue(
		Microsoft::WRL::ComPtr<ID3D12Device2> device, 
		D3D12_COMMAND_LIST_TYPE type);

	Microsoft::WRL::ComPtr<IDXGISwapChain4> CreateSwapChain(HWND hwnd, 
		Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue, 
		uint32_t width, uint32_t height, uint32_t bufferCount);

	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> CreateDescriptorHeap(
		Microsoft::WRL::ComPtr<ID3D12Device> device,
		D3D12_DESCRIPTOR_HEAP_TYPE type,
		uint32_t numDescriptors);

	Microsoft::WRL::ComPtr<ID3D12CommandAllocator> CreateCommandAllocator(
		Microsoft::WRL::ComPtr<ID3D12Device2> device,
		D3D12_COMMAND_LIST_TYPE type);

	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> CreateCommandList(
		Microsoft::WRL::ComPtr<ID3D12Device2> device,
		Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocator,
		D3D12_COMMAND_LIST_TYPE);

	Microsoft::WRL::ComPtr<ID3D12Fence> CreateFence(
		Microsoft::WRL::ComPtr<ID3D12Device2> device);

	HANDLE CreateEventHandle();

	uint64_t Signal(
		Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue,
		Microsoft::WRL::ComPtr<ID3D12Fence> fence,
		uint64_t& fenceValue);

	void WaitForFenceValue(
		Microsoft::WRL::ComPtr<ID3D12Fence> fence,
		uint64_t fenceValue,
		HANDLE fenceEvent);

	void Flush(
		Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue,
		Microsoft::WRL::ComPtr<ID3D12Fence> fence,
		uint64_t& fenceValue, HANDLE fenceEvent);

	void Update();
	void Render();

	void UpdateRenderTargetViews(
		Microsoft::WRL::ComPtr<ID3D12Device2> device,
		Microsoft::WRL::ComPtr<IDXGISwapChain4> swapChain,
		Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptorHeap);

	bool CheckTearingSupport();
	void SetFullscreen(bool fullscreen);

	static const uint8_t m_NumFrames = 3;
	bool m_UseWarp = false;

	uint32_t m_ClientWidth = 1280;
	uint32_t m_ClientHeight = 720;
	
	bool m_IsInitialized = false;

	HWND m_hWnd;
	RECT m_WindowRect;

	Microsoft::WRL::ComPtr<ID3D12Device2> m_Device;
	Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_CommandQueue;
	Microsoft::WRL::ComPtr<IDXGISwapChain4> m_SwapChain;
	Microsoft::WRL::ComPtr<ID3D12Resource> m_BackBuffers[m_NumFrames];
	Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_CommandList;
	Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_CommandAllocator;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_RTVDescriptorHeap;

	UINT m_RTVDescriptorSize;
	UINT m_CurrentBackBufferIndex;

	//Synchronization Objects
	Microsoft::WRL::ComPtr<ID3D12Fence> m_Fence;
	uint64_t m_FenceValue = 0;
	uint64_t m_FrameFenceValues[m_NumFrames] = {};
	HANDLE m_FenceEvent;

	//Presentation Objects
	bool m_VSync = true;
	bool m_TearingSupported = false;
	bool m_Fullscreen = false;

private:

};

