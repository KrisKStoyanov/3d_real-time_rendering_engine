#include "Engine.h"

bool Engine::Initialize(WINDOW_DESC* window_desc, CORE_DESC* core_desc)
{
	m_pWindow = Window::Create(window_desc);
	if (m_pWindow) {
		m_pCore = Core::Create(core_desc, m_pWindow->GetHandle());
		if (m_pCore) {
			m_isRunning = m_pCore->Initialize();
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
		//Use Core to process input, render, etc
		m_pCore->OnFrameRender();
	}
	return (int)msg.wParam;
}

void Engine::Shutdown()
{
	m_pWindow->Shutdown();
	m_pCore->Shutdown();
}
