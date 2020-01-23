#include "Renderer.h"

BOOL Renderer::OnStart(Window* win)
{
	if (win) {
		m_Window = win;
		m_GC = m_Window->GetGraphicsContext();
		return true;
	}
	else {
		return false;
	}
}

void Renderer::OnUpdate()
{
	m_Window->OnUpdate();
}
