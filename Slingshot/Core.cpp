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

Core* Core::Create(HWND hWnd)
{
	return new Core(hWnd);
}

Core::Core(HWND hWnd) : m_hWnd(hWnd), m_pRenderer(nullptr), m_isActive(true)
{
	SetWindowLongPtrW(
		hWnd, GWLP_USERDATA,
		reinterpret_cast<LONG_PTR>(this));
	SetWindowLongPtrW(
		hWnd, GWLP_WNDPROC,
		reinterpret_cast<LONG_PTR>(CoreProc));
}

LRESULT Core::HandleMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg) {
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
		m_isActive = false;
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

bool Core::InitializeRenderer(RENDERER_DESC* renderer_desc)
{
	m_pRenderer = Renderer::Create(m_hWnd, renderer_desc);
	bool success = (m_pRenderer != nullptr);
	if (success) {
		success = m_pRenderer->Initialize();
	}
	return success;
}

bool Core::OnUpdate(Stage* stage)
{
	unsigned int entityCount = stage->GetEntityCount();
	Entity* entityCollection = stage->GetEntityCollection();
	for (unsigned int i = 0; i < entityCount; ++i) {
		Model* model = (entityCollection + i)->GetModel();
		if (model != nullptr) {
			m_pRenderer->OnFrameRender(
				model, 
				(entityCollection + i)->GetTransform(), 
				stage->GetMainCamera());
		}
	}
	return m_isActive;
}

void Core::Shutdown(void)
{
	SAFE_SHUTDOWN(m_pRenderer);
}

Renderer* Core::GetRenderer()
{
	return m_pRenderer;
}
