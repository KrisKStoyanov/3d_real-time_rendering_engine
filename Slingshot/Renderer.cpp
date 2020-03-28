#include "Renderer.h"

Renderer* Renderer::Create(HWND hWnd, RENDERER_DESC& renderer_desc)
{
	return new Renderer(hWnd, renderer_desc);
}

Renderer::Renderer(HWND hWnd, RENDERER_DESC& renderer_desc) : 
	m_pGraphicsContext(nullptr), m_pPipelineState(nullptr)
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

bool Renderer::Initialize()
{
	return (m_pGraphicsContext->Initialize());
}

void Renderer::Draw(Scene& scene)
{
	m_pGraphicsContext->StartFrameRender();
	
	m_pPipelineState->UpdateVSPerFrame(
		DirectX::XMMatrixTranspose(scene.GetCamera(scene.GetMainCameraID())->GetCamera()->GetViewMatrix()),
		DirectX::XMMatrixTranspose(scene.GetCamera(scene.GetMainCameraID())->GetCamera()->GetProjectionMatrix()));
	m_pPipelineState->UpdatePSPerFrame(
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
			(scene.GetEntityCollection() + i)->GetModel()->GetMesh()->GetVertexBuffer()->Bind(*m_pGraphicsContext->GetDeviceContext());
			(scene.GetEntityCollection() + i)->GetModel()->GetMesh()->GetIndexBuffer()->Bind(*m_pGraphicsContext->GetDeviceContext());

			m_pPipelineState->Bind(*m_pGraphicsContext->GetDeviceContext());

			m_pPipelineState->UpdateVSPerEntity(
				DirectX::XMMatrixTranspose((scene.GetEntityCollection() + i)->GetTransform()->GetWorldMatrix()));

			m_pPipelineState->UpdatePSPerEntity(
				(scene.GetEntityCollection() + i)->GetModel()->GetMesh()->GetMaterial()->GetSurfaceColor(),
				(scene.GetEntityCollection() + i)->GetModel()->GetMesh()->GetMaterial()->GetRoughness());
			
			m_pPipelineState->BindConstantBuffers(*m_pGraphicsContext->GetDeviceContext());

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

PipelineState* Renderer::GetPipelineState(ShadingModel shadingModel)
{
	//have a switch statement that returns corresponding pipeline state
	//or loop through an array of pipeline states and match the vertex type with the queried argument

	return m_pPipelineState;
}

void Renderer::SetPipelineState(const PIPELINE_DESC& pipelineDesc)
{
	m_pPipelineState = PipelineState::Create(*m_pGraphicsContext, pipelineDesc);
}
