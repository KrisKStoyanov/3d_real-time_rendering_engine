#include "Core.h"

bool Core::Initialize(WINDOW_DESC * window_desc)
{
	m_pWindow = new Window();
	if (!m_pWindow->Create(window_desc)) {
		return false;
	}
	m_pGC = m_pWindow->GetGraphicsContext();
	if (!m_pGC) {
		return false;
	}
	m_isRunning = true;
	return m_isRunning;
}

int Core::Run()
{
	int status = EXIT_SUCCESS;
	while (m_isRunning) {
		status = m_pWindow->OnUpdate(m_isRunning);
	}
	return status;
}

void Core::Shutdown()
{
	m_pWindow->Shutdown();
}
