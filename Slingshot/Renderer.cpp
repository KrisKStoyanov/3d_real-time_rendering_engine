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
	unsigned int startRenderId = stage.GetStartRenderID();
	for (unsigned int i = 0; i < startRenderId; ++i) {
		(stage.GetEntityCollection() + i)->GetTransform()->OnFrameRender();
	}
	stage.GetMainCamera()->GetCamera()->OnFrameRender(*stage.GetMainCamera()->GetTransform());
	for (unsigned int i = startRenderId; i < stage.GetEntityCount(); ++i)
	{
		Render(
			*stage.GetMainCamera(), 
			*(stage.GetEntityCollection() + i), 
			*(stage.GetEntityCollection() + 1), // overhaul rendering structure to avoid hardcoding & allow for multiple rights
			*GetPipelineState((stage.GetEntityCollection() + i)->GetModel()->GetShadingModel()));
	}
	m_pGraphicsContext->EndFrameRender();
}

void Renderer::Render(Entity& camera, Entity& renderable, Entity& light, PipelineState& pipelineState)
{
	renderable.GetTransform()->OnFrameRender();

	ID3D11DeviceContext* deviceContext = m_pGraphicsContext->GetDeviceContext();
	deviceContext->IASetVertexBuffers(0, 1, 
		renderable.GetModel()->GetMesh()->GetVertexBuffer().GetAddressOf(), 
		renderable.GetModel()->GetMesh()->GetVertexBufferStride(), 
		renderable.GetModel()->GetMesh()->GetVertexBufferOffset());
	deviceContext->IASetIndexBuffer(
		renderable.GetModel()->GetMesh()->GetIndexBuffer().Get(), DXGI_FORMAT_R32_UINT, 0);
	deviceContext->IASetPrimitiveTopology(renderable.GetModel()->GetMesh()->GetTopology());

	deviceContext->IASetInputLayout(pipelineState.GetInputLayout());
	deviceContext->VSSetShader(pipelineState.GetVertexShader(), nullptr, 0);
	deviceContext->PSSetShader(pipelineState.GetPixelShader(), nullptr, 0);

	DirectX::XMMATRIX worldMatrixT = DirectX::XMMatrixTranspose(renderable.GetTransform()->GetWorldMatrix());
	DirectX::XMMATRIX viewMatrixT = DirectX::XMMatrixTranspose(camera.GetCamera()->GetViewMatrix());
	DirectX::XMMATRIX projMatrixT = DirectX::XMMatrixTranspose(camera.GetCamera()->GetProjectionMatrix());

	VS_CONSTANT_BUFFER vs_cb;
	
	vs_cb.worldMatrix = worldMatrixT;
	vs_cb.viewMatrix = viewMatrixT;
	vs_cb.projMatrix = projMatrixT;

	PS_CONSTANT_BUFFER ps_cb;

	ps_cb.camPos = DirectX::XMVector4Transform(camera.GetTransform()->GetPosition(), DirectX::XMMatrixTranspose(camera.GetTransform()->GetWorldMatrix()));
	ps_cb.lightPos = DirectX::XMVector4Transform(light.GetTransform()->GetPosition(), DirectX::XMMatrixTranspose(light.GetTransform()->GetWorldMatrix()));
	ps_cb.lightColor = light.GetLight()->GetColor();
	ps_cb.lightInt = light.GetLight()->GetIntensity();

	deviceContext->UpdateSubresource(renderable.GetModel()->GetMesh()->GetVSCB().Get(), 0, nullptr, &vs_cb, 0, 0);
	deviceContext->VSSetConstantBuffers(0, 1, renderable.GetModel()->GetMesh()->GetVSCB().GetAddressOf());

	deviceContext->UpdateSubresource(renderable.GetModel()->GetMesh()->GetPSCB().Get(), 0, nullptr, &ps_cb, 0, 0);
	deviceContext->PSSetConstantBuffers(0, 1, renderable.GetModel()->GetMesh()->GetPSCB().GetAddressOf());

	deviceContext->DrawIndexed(renderable.GetModel()->GetMesh()->GetIndexCount(), 0, 0);
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
