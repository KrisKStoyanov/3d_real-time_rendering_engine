#include "Renderer.h"

Renderer* Renderer::Create(HWND hWnd, RENDERER_DESC* renderer_desc)
{
	return new Renderer(hWnd, renderer_desc);
}

Renderer::Renderer(HWND hWnd, RENDERER_DESC* renderer_desc) : m_pDesc(renderer_desc), m_pGraphicsContext(nullptr)
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
}

bool Renderer::Initialize()
{
	bool status = m_pGraphicsContext->Initialize();
	return status;
}

void Renderer::OnFrameRender(Model* model)
{
	m_pGraphicsContext->StartFrameRender();
	model->Render(m_pGraphicsContext);
	m_pGraphicsContext->EndFrameRender();
}

void Renderer::Shutdown()
{
	m_pGraphicsContext->Shutdown();
}

D3D11Context* Renderer::GetGraphicsContext()
{
	return m_pGraphicsContext;
}