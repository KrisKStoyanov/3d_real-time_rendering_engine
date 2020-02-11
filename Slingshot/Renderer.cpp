#include "Renderer.h"

Renderer* Renderer::Create(HWND hWnd, RENDERER_DESC* renderer_desc)
{
	return new Renderer(hWnd, renderer_desc);
}

Renderer::Renderer(HWND hWnd, RENDERER_DESC* renderer_desc) : m_pDesc(nullptr), m_pGraphicsContext(nullptr)
{
	switch (renderer_desc->graphicsContextType)
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
		return;
	}
	m_pDesc = renderer_desc;
}

bool Renderer::Initialize()
{
	bool status = m_pGraphicsContext->Initialize();
	return status;
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

