#include "D3D12GraphicsContext.h"

D3D12GraphicsContext::D3D12GraphicsContext() {

}

D3D12GraphicsContext::~D3D12GraphicsContext()
{
}

LRESULT D3D12GraphicsContext::HandleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	return LRESULT();
}

void D3D12GraphicsContext::OnCreate(HWND hwnd)
{
}

void D3D12GraphicsContext::OnDestroy()
{
}

void D3D12GraphicsContext::OnPaint()
{
}

void D3D12GraphicsContext::OnResize(uint32_t width, uint32_t height)
{
}

void D3D12GraphicsContext::OnLButtonDown(int pixelX, int pixelY, DWORD flags)
{
}

void D3D12GraphicsContext::OnLButtonUp()
{
}

void D3D12GraphicsContext::OnMouseMove(int pixelX, int pixelY, DWORD flags)
{
}

void D3D12GraphicsContext::ParseCommandLineArguments()
{
}

void D3D12GraphicsContext::EnableDebugLayer()
{
}

Microsoft::WRL::ComPtr<IDXGIAdapter4> D3D12GraphicsContext::GetAdapter(bool useWarp)
{
	return Microsoft::WRL::ComPtr<IDXGIAdapter4>();
}

Microsoft::WRL::ComPtr<ID3D12Device2> D3D12GraphicsContext::CreateDevice(Microsoft::WRL::ComPtr<IDXGIAdapter4> adapter)
{
	return Microsoft::WRL::ComPtr<ID3D12Device2>();
}

Microsoft::WRL::ComPtr<ID3D12CommandQueue> D3D12GraphicsContext::CreateCommandQueue(Microsoft::WRL::ComPtr<ID3D12Device2> device, D3D12_COMMAND_LIST_TYPE type)
{
	return Microsoft::WRL::ComPtr<ID3D12CommandQueue>();
}

Microsoft::WRL::ComPtr<IDXGISwapChain4> D3D12GraphicsContext::CreateSwapChain(HWND hwnd, Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue, uint32_t width, uint32_t height, uint32_t bufferCount)
{
	return Microsoft::WRL::ComPtr<IDXGISwapChain4>();
}

Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> D3D12GraphicsContext::CreateDescriptorHeap(Microsoft::WRL::ComPtr<ID3D12Device> device, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t numDescriptors)
{
	return Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>();
}

Microsoft::WRL::ComPtr<ID3D12CommandAllocator> D3D12GraphicsContext::CreateCommandAllocator(Microsoft::WRL::ComPtr<ID3D12Device2> device, D3D12_COMMAND_LIST_TYPE type)
{
	return Microsoft::WRL::ComPtr<ID3D12CommandAllocator>();
}

Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> D3D12GraphicsContext::CreateCommandList(Microsoft::WRL::ComPtr<ID3D12Device2> device, Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocator, D3D12_COMMAND_LIST_TYPE)
{
	return Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>();
}

Microsoft::WRL::ComPtr<ID3D12Fence> D3D12GraphicsContext::CreateFence(Microsoft::WRL::ComPtr<ID3D12Device2> device)
{
	return Microsoft::WRL::ComPtr<ID3D12Fence>();
}

HANDLE D3D12GraphicsContext::CreateEventHandle()
{
	return HANDLE();
}

uint64_t D3D12GraphicsContext::Signal(Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue, Microsoft::WRL::ComPtr<ID3D12Fence> fence, uint64_t& fenceValue)
{
	return uint64_t();
}

void D3D12GraphicsContext::WaitForFenceValue(Microsoft::WRL::ComPtr<ID3D12Fence> fence, uint64_t fenceValue, HANDLE fenceEvent)
{
}

void D3D12GraphicsContext::Flush(Microsoft::WRL::ComPtr<ID3D12CommandQueue> commandQueue, Microsoft::WRL::ComPtr<ID3D12Fence> fence, uint64_t& fenceValue, HANDLE fenceEvent)
{
}

void D3D12GraphicsContext::Update()
{
}

void D3D12GraphicsContext::Render()
{
}

void D3D12GraphicsContext::UpdateRenderTargetViews(Microsoft::WRL::ComPtr<ID3D12Device2> device, Microsoft::WRL::ComPtr<IDXGISwapChain4> swapChain, Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptorHeap)
{
}

bool D3D12GraphicsContext::CheckTearingSupport()
{
	return false;
}

void D3D12GraphicsContext::SetFullscreen(bool fullscreen)
{
}
