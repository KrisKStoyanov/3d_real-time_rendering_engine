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
}

bool Renderer::Initialize(PIPELINE_DESC pipeline_desc)
{
	if (!m_pGraphicsContext->Initialize())
	{
		return false;
	}
	m_pNvExtension = NvExtension::Create();
	if (!m_pNvExtension->Initialize(*m_pGraphicsContext))
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

	DirectX::XMMATRIX lightViewMatrix = DirectX::XMMatrixTranspose(scene.GetLights()->GetTransform()->GetViewMatrix());
	DirectX::XMMATRIX lightProjMatrix = DirectX::XMMatrixTranspose(DirectX::XMMatrixPerspectiveFovLH(
		DirectX::XMConvertToRadians(90.0f),
		1.0f,
		1.0f,
		1000.0f)); //represent point light influence radius

	// Depth Pre-pass
	m_pDepthMap->UpdatePerFrame(*m_pGraphicsContext->GetContext());

	PerFrameDataVS_DM perFrameDataVS_DM;
	perFrameDataVS_DM.viewMatrix = lightViewMatrix;
	perFrameDataVS_DM.projectionMatrix = lightProjMatrix;
	
	m_pDepthMap->UpdateBuffersPerFrame(perFrameDataVS_DM);

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
	perFrameDataVS.lightViewMatrix = lightViewMatrix;
	perFrameDataVS.lightProjMatrix = lightProjMatrix;
	perFrameDataVS.camPos = scene.GetCamera(scene.GetMainCameraID())->GetTransform()->GetPosition(); // may need conversion to world space
	perFrameDataVS.lightPos = scene.GetLights()->GetTransform()->GetPosition(); // may need conversion to world space

	PerFrameDataPS_DI perFrameDataPS;
	perFrameDataPS.camPos = DirectX::XMVector4Transform(
		scene.GetCamera(scene.GetMainCameraID())->GetTransform()->GetPosition(),
		DirectX::XMMatrixTranspose(scene.GetCamera(scene.GetMainCameraID())->GetTransform()->GetWorldMatrix()));
	perFrameDataPS.lightPos = DirectX::XMVector4Transform(
		scene.GetLights()->GetTransform()->GetPosition(),
		DirectX::XMMatrixTranspose((scene.GetEntityCollection() + 1)->GetTransform()->GetWorldMatrix()));
	perFrameDataPS.ambientColor = DirectX::XMFLOAT4(0.5f, 0.5f, 0.5f, 1.0f);
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

void Renderer::UpdateConstantVRS()
{
	m_pNvExtension->SetConstantVRS(true, *m_pGraphicsContext);
}

void Renderer::ToggleConstantVRS()
{
	m_pNvExtension->SetConstantVRS(!m_pNvExtension->GetVRS(), *m_pGraphicsContext);
}

void Renderer::ToggleVRSwithSRS()
{
	m_pNvExtension->SetVRSwithSRS(!m_pNvExtension->GetVRS(), *m_pGraphicsContext);
}

GraphicsContext* Renderer::GetGraphicsContext()
{
	return m_pGraphicsContext;
}

void Renderer::SetupPhotonMap(GEOMETRY_DESC& desc)
{
	m_pPhotonMap = PhotonMap::Create(desc);
}
