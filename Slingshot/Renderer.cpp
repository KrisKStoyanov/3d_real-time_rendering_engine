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
	m_pGraphicsContext->UpdatePerConfig(*m_pPipelineState);
	return (m_pPipelineState != nullptr);
}

void Renderer::Draw(Scene& scene)
{
	m_pGraphicsContext->UpdatePerFrame(*m_pPipelineState);
	m_pGraphicsContext->SetBackBufferRender(*m_pPipelineState);
	//m_pGraphicsContext->SetShadowMapRender(*m_pPipelineState);

	PerFrameDataVS perFrameDataVS;
	perFrameDataVS.cameraViewMatrix = DirectX::XMMatrixTranspose(scene.GetCamera(scene.GetMainCameraID())->GetCamera()->GetViewMatrix());
	perFrameDataVS.cameraProjMatrix = DirectX::XMMatrixTranspose(scene.GetCamera(scene.GetMainCameraID())->GetCamera()->GetProjectionMatrix());
	perFrameDataVS.lightViewMatrix = scene.GetLights()->GetTransform()->GetViewMatrix();
	perFrameDataVS.lightProjMatrix = scene.GetLights()->GetTransform()->GetProjectionMatrix();
	perFrameDataVS.lightPos = scene.GetLights()->GetTransform()->GetPosition();

	m_pPipelineState->UpdateVSPerFrame(perFrameDataVS);

	PerFrameDataPS perFrameDataPS;
	perFrameDataPS.ambientColor = DirectX::XMFLOAT4(0.4f, 0.4f, 0.4f, 1.0f);
	perFrameDataPS.diffuseColor = scene.GetLights()->GetLight()->GetColor();

	m_pPipelineState->UpdatePSPerFrame(perFrameDataPS);

	for (int i = 0; i < scene.GetEntityCount(); ++i)
	{
		Entity& entity = *(scene.GetEntityCollection() + i);
		if(entity.GetModel())
		{
			m_pGraphicsContext->BindMeshBuffers(
				*static_cast<D3D11VertexBuffer*>(entity.GetModel()->GetMesh()->GetVertexBuffer()),
				*static_cast<D3D11IndexBuffer*>(entity.GetModel()->GetMesh()->GetIndexBuffer()));

			PerDrawCallDataVS perDrawCallDataVS;
			perDrawCallDataVS.worldMatrix = DirectX::XMMatrixTranspose(entity.GetTransform()->GetWorldMatrix());			
			m_pPipelineState->UpdateVSPerDrawCall(perDrawCallDataVS);

			PerDrawCallDataPS perDrawCallDataPS;
			perDrawCallDataPS.surfaceColor = entity.GetModel()->GetMesh()->GetMaterial()->GetSurfaceColor();
			m_pPipelineState->UpdatePSPerDrawCall(perDrawCallDataPS);
			
			m_pGraphicsContext->BindConstantBuffers(*m_pPipelineState);

			m_pGraphicsContext->DrawIndexed(entity.GetModel()->GetMesh()->GetIndexBuffer()->GetElementCount(), 0, 0);
		}
	}

	//m_pGraphicsContext->SetBackBufferRender(*m_pPipelineState);

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

void Renderer::SetupPhotonMap(GEOMETRY_DESC& desc)
{
	m_pPhotonMap = PhotonMap::Create(desc);
}
