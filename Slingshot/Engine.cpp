#include "Engine.h"

bool Engine::Initialize(WINDOW_DESC* window_desc, CORE_DESC* core_desc)
{
	m_pWindow = Window::Create(window_desc, core_desc);
	if (m_pWindow) {
		m_pCore = m_pWindow->GetMessageHandler();
		if (m_pCore) {
			m_isRunning = true;
		}
	}
	return m_isRunning;
}

int Engine::Run()
{
	int status = EXIT_SUCCESS;
	while (m_isRunning) {
		status = m_pWindow->OnUpdate(m_isRunning);
		//Do stuff with Core here
	}
	return status;
}

void Engine::Shutdown()
{
	m_pWindow->Shutdown();
	m_pCore->Shutdown();
}
