#include "Renderer.h"

Renderer* Renderer::Create(HWND hWnd, RENDERER_DESC& renderer_desc)
{
	return new Renderer(hWnd, renderer_desc);
}

Renderer::Renderer(HWND hWnd, RENDERER_DESC& renderer_desc) : 
	m_pGraphicsContext(nullptr)
{
	switch (renderer_desc.gfxContextType)
	{
	case ContextType::D3D11:
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

bool Renderer::Initialize(PIPELINE_DESC pipeline_desc)
{
	if (!m_pGraphicsContext->Initialize())
	{
		return false;
	}
	m_pPipelineState = m_pGraphicsContext->CreatePipelineState(pipeline_desc);
	return (m_pPipelineState != nullptr);
}

void Renderer::Draw(Scene& scene)
{
	m_pGraphicsContext->StartFrameRender();
	m_pGraphicsContext->BindPipelineState(*m_pPipelineState);

	m_pPipelineState->UpdatePerFrame(
		DirectX::XMMatrixTranspose(scene.GetCamera(scene.GetMainCameraID())->GetCamera()->GetViewMatrix()),
		DirectX::XMMatrixTranspose(scene.GetCamera(scene.GetMainCameraID())->GetCamera()->GetProjectionMatrix()),
		DirectX::XMVector4Transform(
			scene.GetCamera(scene.GetMainCameraID())->GetTransform()->GetPosition(),
			DirectX::XMMatrixTranspose(scene.GetCamera(scene.GetMainCameraID())->GetTransform()->GetWorldMatrix())),
		DirectX::XMVector4Transform(
		(scene.GetEntityCollection() + 1)->GetTransform()->GetPosition(),
			DirectX::XMMatrixTranspose((scene.GetEntityCollection() + 1)->GetTransform()->GetWorldMatrix())),
			(scene.GetEntityCollection() + 1)->GetLight()->GetColor());


	for (int i = 0; i < scene.GetEntityCount(); ++i)
	{
		Entity& entity = *(scene.GetEntityCollection() + i);
		if(entity.GetModel())
		{
			m_pGraphicsContext->BindMeshBuffers(
				*static_cast<D3D11VertexBuffer*>(entity.GetModel()->GetMesh()->GetVertexBuffer()),
				*static_cast<D3D11IndexBuffer*>(entity.GetModel()->GetMesh()->GetIndexBuffer()));

			m_pPipelineState->UpdatePerModel(
				DirectX::XMMatrixTranspose(entity.GetTransform()->GetWorldMatrix()),
				entity.GetModel()->GetMesh()->GetMaterial()->GetSurfaceColor(),
				entity.GetModel()->GetMesh()->GetMaterial()->GetRoughness());
			
			m_pGraphicsContext->BindConstantBuffers(*m_pPipelineState);

			//Forward implementation
			m_pGraphicsContext->DrawIndexed(entity.GetModel()->GetMesh()->GetIndexBuffer()->GetElementCount(), 0, 0);
		}
	}

	m_pGraphicsContext->EndFrameRender();
}

void Renderer::Shutdown()
{
	m_pPipelineState->Shutdown();
	m_pGraphicsContext->Shutdown();
}

GraphicsContext* Renderer::GetGraphicsContext()
{
	return m_pGraphicsContext;
}
