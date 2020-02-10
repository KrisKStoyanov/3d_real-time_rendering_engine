#include "Core.h"

LRESULT CALLBACK CoreProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	Core* pCore =
		reinterpret_cast<Core*>
		(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

	if (pCore) {
		return pCore->HandleMessage(hwnd, uMsg, wParam, lParam);
	}
	else {
		DestroyWindow(hwnd);
		PostQuitMessage(0);
	}
	return DefWindowProcW(hwnd, uMsg, wParam, lParam);
}

Core* Core::Create(CORE_DESC* core_desc, HWND hWnd)
{
	return new Core(core_desc, hWnd);
}

Core::Core(CORE_DESC* core_desc, HWND hWnd) : m_pDesc(nullptr), m_pGraphicsContext(nullptr)
{
	SetWindowLongPtrW(
		hWnd, GWLP_USERDATA,
		reinterpret_cast<LONG_PTR>(this));
	SetWindowLongPtrW(
		hWnd, GWLP_WNDPROC,
		reinterpret_cast<LONG_PTR>(CoreProc));

	switch (core_desc->graphicsContextType)
	{
	case GraphicsContextType::D3D11:
	{
		m_pGraphicsContext = D3D11Context::Create(hWnd);
	}
	break;
	default:
	{
		m_pGraphicsContext = D3D11Context::Create(hWnd);
	}
	break;
	}
	m_pDesc = core_desc;
}

LRESULT Core::HandleMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg) {
	case WM_PAINT:
	{
		OnFrameRender();
	}
	break;
	case WM_SYSKEYDOWN:
	case WM_KEYDOWN:
	{
		switch (wParam) {
		case VK_ESCAPE:
		{
			DestroyWindow(hWnd);
		}
		}
	}
	break;
	case WM_QUIT:
	case WM_DESTROY:
	{
		PostQuitMessage(0);
	}
	break;
	default:
	{
		DefWindowProcW(hWnd, uMsg, wParam, lParam);
	}
	break;
	}
	return 0;
}

bool Core::Initialize()
{
	bool success = m_pGraphicsContext->Initialize();
	return success;
}

void Core::OnFrameRender(void)
{
	m_pGraphicsContext->StartFrameRender();
	//Render something
	m_pGraphicsContext->EndFrameRender();
}

void Core::Shutdown()
{
	m_pGraphicsContext->Shutdown();
}
