#include "Renderer.h"

Renderer* Renderer::Create(HWND hWnd, RENDERER_DESC& renderer_desc)
{
	return new Renderer(hWnd, renderer_desc);
}

Renderer::Renderer(HWND hWnd, RENDERER_DESC& renderer_desc) : m_pGraphicsContext(nullptr)
{
	switch (renderer_desc.graphicsContextType)
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
	return (m_pGraphicsContext->Initialize());
}

void Renderer::OnFrameRender(Stage& stage)
{
	m_pGraphicsContext->StartFrameRender();
	stage.GetMainCamera()->GetTransform()->OnFrameRender();
	stage.GetMainCamera()->GetCamera()->OnFrameRender(*stage.GetMainCamera()->GetTransform());
	for (unsigned int i = 0; i < stage.GetEntityCount(); ++i) {
		(stage.GetEntityCollection() + i)->GetTransform()->OnFrameRender();
		Model* model = (stage.GetEntityCollection() + i)->GetModel();
		if (model != nullptr) {
			model->OnFrameRender(*m_pGraphicsContext, 
				DirectX::XMMatrixTranspose(
					(stage.GetEntityCollection() + i)->GetTransform()->GetWorldMatrix() *
					stage.GetMainCamera()->GetCamera()->GetViewMatrix() *
					stage.GetMainCamera()->GetCamera()->GetProjectionMatrix()));
		}
	}
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