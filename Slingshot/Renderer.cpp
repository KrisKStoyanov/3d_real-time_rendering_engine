#include "Renderer.h"

Renderer* Renderer::Create(HWND hWnd, RENDERER_DESC& renderer_desc)
{
	return new Renderer(hWnd, renderer_desc);
}

Renderer::Renderer(HWND hWnd, RENDERER_DESC& renderer_desc) : 
	m_pGraphicsContext(nullptr), m_pPipelineState(nullptr)
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
	ID3D11DeviceContext* deviceContext = m_pGraphicsContext->GetDeviceContext();

	unsigned int entityCount = stage.GetEntityCount();
	stage.GetMainCamera()->GetTransform()->OnFrameRender();
	stage.GetMainCamera()->GetCamera()->OnFrameRender(*stage.GetMainCamera()->GetTransform());

	VS_CONSTANT_BUFFER vs_cb;
	vs_cb.viewMatrix = DirectX::XMMatrixTranspose(stage.GetMainCamera()->GetCamera()->GetViewMatrix());
	vs_cb.projMatrix = DirectX::XMMatrixTranspose(stage.GetMainCamera()->GetCamera()->GetProjectionMatrix());
	PS_CONSTANT_BUFFER ps_cb;
	ps_cb.camPos = DirectX::XMVector4Transform(
		stage.GetMainCamera()->GetTransform()->GetPosition(), DirectX::XMMatrixTranspose(stage.GetMainCamera()->GetTransform()->GetWorldMatrix()));
	ps_cb.lightPos = DirectX::XMVector4Transform(
		(stage.GetEntityCollection() + 1)->GetTransform()->GetPosition(), 
		DirectX::XMMatrixTranspose((stage.GetEntityCollection() + 1)->GetTransform()->GetWorldMatrix()));
	ps_cb.lightColor = (stage.GetEntityCollection() + 1)->GetLight()->GetColor();

	for (unsigned int i = 0; i < entityCount; ++i) 
	{
		(stage.GetEntityCollection() + i)->GetTransform()->OnFrameRender();
		if((stage.GetEntityCollection() + i)->GetModel())
		{
			deviceContext->IASetVertexBuffers(0, 1,
				(stage.GetEntityCollection() + i)->GetModel()->GetMesh()->GetVertexBuffer().GetAddressOf(),
				(stage.GetEntityCollection() + i)->GetModel()->GetMesh()->GetVertexBufferStride(),
				(stage.GetEntityCollection() + i)->GetModel()->GetMesh()->GetVertexBufferOffset());
			deviceContext->IASetIndexBuffer(
				(stage.GetEntityCollection() + i)->GetModel()->GetMesh()->GetIndexBuffer().Get(), DXGI_FORMAT_R32_UINT, 0);
			deviceContext->IASetPrimitiveTopology((stage.GetEntityCollection() + i)->GetModel()->GetMesh()->GetTopology());

			deviceContext->IASetInputLayout(m_pPipelineState->GetInputLayout());
			deviceContext->VSSetShader(m_pPipelineState->GetVertexShader(), nullptr, 0);
			deviceContext->PSSetShader(m_pPipelineState->GetPixelShader(), nullptr, 0);

			vs_cb.worldMatrix = DirectX::XMMatrixTranspose((stage.GetEntityCollection() + i)->GetTransform()->GetWorldMatrix());

			ps_cb.surfaceColor = (stage.GetEntityCollection() + i)->GetModel()->GetMesh()->GetMaterial()->GetSurfaceColor();

			PipelineState pipelineState = *GetPipelineState((stage.GetEntityCollection() + i)->GetModel()->GetMesh()->GetMaterial()->GetShadingModel());

			deviceContext->UpdateSubresource(pipelineState.GetVSCB().Get(), 0, nullptr, &vs_cb, 0, 0);
			deviceContext->VSSetConstantBuffers(0, 1, pipelineState.GetVSCB().GetAddressOf());
			deviceContext->UpdateSubresource(pipelineState.GetPSCB().Get(), 0, nullptr, &ps_cb, 0, 0);
			deviceContext->PSSetConstantBuffers(0, 1, pipelineState.GetPSCB().GetAddressOf());

			deviceContext->DrawIndexed((stage.GetEntityCollection() + i)->GetModel()->GetMesh()->GetIndexCount(), 0, 0);
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

void Renderer::SetPipelineState(PIPELINE_DESC pipeline_desc, ShadingModel shadingModel)
{
	m_pPipelineState = PipelineState::Create(*m_pGraphicsContext, pipeline_desc, shadingModel);
}
