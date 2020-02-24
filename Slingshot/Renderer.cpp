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
	for (unsigned int i = 1; i < stage.GetEntityCount(); ++i) //Begin from 1 to exclude camera entity
	{
		Render(
			*stage.GetMainCamera()->GetCamera(), 
			*(stage.GetEntityCollection() + i)->GetTransform(), 
			*(stage.GetEntityCollection() + i)->GetModel()->GetMesh(),
			*(stage.GetEntityCollection() + i)->GetModel()->GetGraphicsProps());
	}
	m_pGraphicsContext->EndFrameRender();
}

void Renderer::Render(Camera& camera, Transform& transform, Mesh& mesh, GraphicsProps& props)
{
	transform.OnFrameRender();

	DirectX::XMMATRIX wvpMatrix = DirectX::XMMatrixTranspose(
		(transform.GetWorldMatrix() *
		camera.GetViewMatrix() *
		camera.GetProjectionMatrix()));

	ID3D11DeviceContext* deviceContext = m_pGraphicsContext->GetDeviceContext();
	deviceContext->IASetVertexBuffers(0, 1, mesh.GetVertexBuffer().GetAddressOf(), mesh.GetVertexBufferStride(), mesh.GetVertexBufferOffset());
	deviceContext->IASetIndexBuffer(mesh.GetIndexBuffer().Get(), DXGI_FORMAT_R32_UINT, 0);
	deviceContext->IASetPrimitiveTopology(mesh.GetTopology());

	deviceContext->IASetInputLayout(props.GetInputLayout());
	deviceContext->VSSetShader(props.GetVertexShader(), nullptr, 0);
	deviceContext->PSSetShader(props.GetPixelShader(), nullptr, 0);

	VS_CONSTANT_BUFFER vs_cb;
	vs_cb.wvpMatrix = wvpMatrix;

	deviceContext->UpdateSubresource(mesh.GetVSCB().Get(), 0, nullptr, &vs_cb, 0, 0);
	deviceContext->VSSetConstantBuffers(0, 1, mesh.GetVSCB().GetAddressOf());

	deviceContext->DrawIndexed(mesh.GetIndexCount(), 0, 0);
}

void Renderer::Shutdown()
{
	m_pGraphicsContext->Shutdown();
}

D3D11Context* Renderer::GetGraphicsContext()
{
	return m_pGraphicsContext;
}