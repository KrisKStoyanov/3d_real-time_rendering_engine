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
	case gfx::ContextType::D3D11:
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
	return (m_pGraphicsContext->Initialize(pipeline_desc));
}

void Renderer::Draw(Scene& scene)
{
	m_pGraphicsContext->StartFrameRender();
	m_pGraphicsContext->BindPipelineState(ShadingModel::GoochShading);

	m_pGraphicsContext->UpdatePipelinePerFrame(
		DirectX::XMMatrixTranspose(scene.GetCamera(scene.GetMainCameraID())->GetCamera()->GetViewMatrix()),
		DirectX::XMMatrixTranspose(scene.GetCamera(scene.GetMainCameraID())->GetCamera()->GetProjectionMatrix()),
		DirectX::XMVector4Transform(
			scene.GetCamera(scene.GetMainCameraID())->GetTransform()->GetPosition(),
			DirectX::XMMatrixTranspose(scene.GetCamera(scene.GetMainCameraID())->GetTransform()->GetWorldMatrix())),
		DirectX::XMVector4Transform(
		(scene.GetEntityCollection() + 1)->GetTransform()->GetPosition(),
			DirectX::XMMatrixTranspose((scene.GetEntityCollection() + 1)->GetTransform()->GetWorldMatrix())),
			(scene.GetEntityCollection() + 1)->GetLight()->GetColor());

	for (unsigned int i = 0; i < scene.GetEntityCount(); ++i)
	{
		if((scene.GetEntityCollection() + i)->GetModel())
		{
			m_pGraphicsContext->BindMeshBuffers(
				*(scene.GetEntityCollection() + i)->GetModel()->GetMesh()->GetVertexBuffer(),
				*(scene.GetEntityCollection() + i)->GetModel()->GetMesh()->GetIndexBuffer());

			m_pGraphicsContext->UpdatePipelinePerModel(
				DirectX::XMMatrixTranspose((scene.GetEntityCollection() + i)->GetTransform()->GetWorldMatrix()),
				(scene.GetEntityCollection() + i)->GetModel()->GetMesh()->GetMaterial()->GetSurfaceColor(),
				(scene.GetEntityCollection() + i)->GetModel()->GetMesh()->GetMaterial()->GetRoughness());
			
			m_pGraphicsContext->BindConstantBuffers();

			//Forward implementation
			m_pGraphicsContext->DrawIndexed((scene.GetEntityCollection() + i)->GetModel()->GetMesh()->GetIndexBuffer()->GetIndexCount(), 0, 0);
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
