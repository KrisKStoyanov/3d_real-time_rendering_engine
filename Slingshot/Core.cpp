#include "Core.h"

Core* Core::Create(CORE_DESC* core_desc, HWND hWnd)
{
	return new Core(core_desc, hWnd);
}

Core::Core(CORE_DESC* core_desc, HWND hWnd) : m_pDesc(nullptr), m_pGraphicsContext(nullptr)
{
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
		//Render
		m_pGraphicsContext->StartFrameRender();
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
	case WM_SYSCHAR:
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

void Core::Shutdown()
{
	m_pGraphicsContext->Shutdown();
}
