#include "Engine.h"

bool Engine::Initialize(WINDOW_DESC* window_desc, CORE_DESC* core_desc)
{
	if (m_pWindow = Window::Create(window_desc)) {
		HWND hWnd = m_pWindow->GetHandle();
		if (m_pCore = Core::Create(core_desc, hWnd)) {
			m_isRunning = m_pCore->Initialize(hWnd);
		}
	}
	return m_isRunning;
}

int Engine::Run()
{
	MSG msg = {};
	while (m_isRunning) {
		while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessageW(&msg);
			if (msg.message == WM_QUIT) {
				m_isRunning = false;
			}
		}
		m_pCore->OnUpdate();
	}
	return (int)msg.wParam;
}

void Engine::Shutdown()
{
	SAFE_SHUTDOWN(m_pCore);
	SAFE_SHUTDOWN(m_pWindow);
}
