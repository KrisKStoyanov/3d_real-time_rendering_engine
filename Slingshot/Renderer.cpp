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
	m_pDepthMap = D3D11DepthMap::Create(*m_pGraphicsContext);
	m_pDirectIllumination = D3D11DirectIllumination::Create(*m_pGraphicsContext);

	return (m_pDepthMap != nullptr && m_pDirectIllumination != nullptr);
}

void Renderer::Draw(Scene& scene)
{
	DirectX::XMMATRIX cameraViewMatrix = DirectX::XMMatrixTranspose(scene.GetCamera(scene.GetMainCameraID())->GetTransform()->GetViewMatrix());
	DirectX::XMMATRIX cameraProjMatrix = DirectX::XMMatrixTranspose(scene.GetCamera(scene.GetMainCameraID())->GetCamera()->GetProjectionMatrix());

	// Depth Pre-pass
	m_pDepthMap->UpdatePerFrame(*m_pGraphicsContext->GetContext());

	DirectX::XMMATRIX lightFrontView;
	DirectX::XMMATRIX lightBackView;
	DirectX::XMMATRIX lightLeftView;
	DirectX::XMMATRIX lightRightView;
	DirectX::XMMATRIX lightTopView;
	DirectX::XMMATRIX lightBottomView;

	scene.GetLights()->GetTransform()->GeneratePanoramicView(
		lightFrontView, 
		lightBackView, 
		lightLeftView, 
		lightRightView, 
		lightTopView, 
		lightBottomView);

	lightFrontView = DirectX::XMMatrixTranspose(lightFrontView);
	lightBackView = DirectX::XMMatrixTranspose(lightBackView);
	lightLeftView = DirectX::XMMatrixTranspose(lightLeftView);
	lightRightView = DirectX::XMMatrixTranspose(lightRightView);
	lightTopView = DirectX::XMMatrixTranspose(lightTopView);
	lightBottomView = DirectX::XMMatrixTranspose(lightBottomView);

	PerFrameDataGS_DM perFrameDataGS_DM;
	perFrameDataGS_DM.viewMatrix0 = lightFrontView;
	perFrameDataGS_DM.viewMatrix1 = lightBackView;
	perFrameDataGS_DM.viewMatrix2 = lightLeftView;
	perFrameDataGS_DM.viewMatrix3 = lightRightView;
	perFrameDataGS_DM.viewMatrix4 = lightTopView;
	perFrameDataGS_DM.viewMatrix5 = lightBottomView;
	perFrameDataGS_DM.projectionMatrix = cameraProjMatrix;

	m_pDepthMap->UpdateBuffersPerFrame(perFrameDataGS_DM);

	for (int i = 0; i < scene.GetEntityCount(); ++i)
	{
		Entity& entity = *(scene.GetEntityCollection() + i);
		if (entity.GetModel())
		{
			m_pGraphicsContext->BindMeshBuffers(
				*static_cast<D3D11VertexBuffer*>(entity.GetModel()->GetMesh()->GetVertexBuffer()),
				*static_cast<D3D11IndexBuffer*>(entity.GetModel()->GetMesh()->GetIndexBuffer()));

			PerDrawCallDataVS_DM perDrawCallDataVS;
			perDrawCallDataVS.worldMatrix = DirectX::XMMatrixTranspose(entity.GetTransform()->GetWorldMatrix());

			m_pDepthMap->UpdateBuffersPerDrawCall(perDrawCallDataVS);
			m_pDepthMap->BindConstantBuffers(*m_pGraphicsContext->GetContext());

			m_pGraphicsContext->DrawIndexed(entity.GetModel()->GetMesh()->GetIndexBuffer()->GetElementCount(), 0, 0);
		}
	}

	// Direct Illumination
	m_pGraphicsContext->SetBackBufferRender();

	m_pDirectIllumination->UpdatePerFrame(*m_pGraphicsContext->GetContext(), m_pDepthMap->GetShaderResourceView());

	PerFrameDataVS_DI perFrameDataVS;
	perFrameDataVS.cameraViewMatrix = cameraViewMatrix;
	perFrameDataVS.cameraProjMatrix = cameraProjMatrix;
	perFrameDataVS.lightViewMatrix = lightFrontView;
	perFrameDataVS.lightProjMatrix = cameraProjMatrix;
	perFrameDataVS.lightPos = scene.GetLights()->GetTransform()->GetPosition();

	PerFrameDataPS_DI perFrameDataPS;
	perFrameDataPS.ambientColor = DirectX::XMFLOAT4(0.4f, 0.4f, 0.4f, 1.0f);
	perFrameDataPS.diffuseColor = scene.GetLights()->GetLight()->GetColor();

	m_pDirectIllumination->UpdateBuffersPerFrame(perFrameDataVS, perFrameDataPS);

	for (int i = 0; i < scene.GetEntityCount(); ++i)
	{
		Entity& entity = *(scene.GetEntityCollection() + i);
		if(entity.GetModel())
		{
			m_pGraphicsContext->BindMeshBuffers(
				*static_cast<D3D11VertexBuffer*>(entity.GetModel()->GetMesh()->GetVertexBuffer()),
				*static_cast<D3D11IndexBuffer*>(entity.GetModel()->GetMesh()->GetIndexBuffer()));

			PerDrawCallDataVS_DI perDrawCallDataVS;
			perDrawCallDataVS.worldMatrix = DirectX::XMMatrixTranspose(entity.GetTransform()->GetWorldMatrix());			

			PerDrawCallDataPS_DI perDrawCallDataPS;
			perDrawCallDataPS.surfaceColor = entity.GetModel()->GetMesh()->GetMaterial()->GetSurfaceColor();

			m_pDirectIllumination->UpdateBuffersPerDrawCall(perDrawCallDataVS, perDrawCallDataPS);
			m_pDirectIllumination->BindConstantBuffers(*m_pGraphicsContext->GetContext());

			m_pGraphicsContext->DrawIndexed(entity.GetModel()->GetMesh()->GetIndexBuffer()->GetElementCount(), 0, 0);
		}
	}

	m_pDirectIllumination->EndFrameRender(*m_pGraphicsContext->GetContext());

	m_pGraphicsContext->EndFrameRender();
}

void Renderer::Shutdown()
{
	SAFE_SHUTDOWN(m_pDepthMap);
	SAFE_SHUTDOWN(m_pDirectIllumination);
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
