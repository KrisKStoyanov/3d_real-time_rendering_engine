#include "Renderer.h"

bool Renderer::Initialize(HWND hWnd, GraphicsContextType graphicsContextType)
{
	switch (graphicsContextType)
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

	if (!m_pGraphicsContext) {
		return false;
	}
	return true;
}

void Renderer::OnFrameRender()
{
	m_pGraphicsContext->StartFrameRender();
	//Render something
	m_pGraphicsContext->EndFrameRender();
}

void Renderer::Shutdown()
{
	m_pGraphicsContext->Shutdown();
}
