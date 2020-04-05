#pragma once
#include "GraphicsContext.h"
#include "D3D11PipelineState.h"

#include <dxgi1_6.h>
#include <dxgidebug.h>

#include "nvapi.h"

class D3D11Context : public GraphicsContext
{
public:
	static D3D11Context * Create(HWND hWnd);
	virtual bool Initialize() override;
	virtual void Shutdown() override;

	void SetShadowMapRender(D3D11PipelineState& pipelineState);
	void SetBackBufferRender(D3D11PipelineState& pipelineState);
	void EndFrameRender();
	
	void BindMeshBuffers(
		D3D11VertexBuffer& vertexBuffer, 
		D3D11IndexBuffer& indexBuffer);

	void BindPipelineState(
		D3D11PipelineState& pipelineState);

	void BindConstantBuffers(
		D3D11PipelineState& pipelineState);

	virtual void DrawIndexed(
		unsigned int indexCount, 
		unsigned int startIndex, 
		unsigned int baseVertexLocation) override;

	virtual D3D11PipelineState* CreatePipelineState(PIPELINE_DESC desc) override;
	virtual D3D11VertexBuffer* CreateVertexBuffer(VERTEX_BUFFER_DESC desc) override;
	virtual D3D11IndexBuffer* CreateIndexBuffer(INDEX_BUFFER_DESC desc) override;
	virtual D3D11ConstantBuffer* CreateConstantBuffer(CONSTANT_BUFFER_DESC desc) override;

	void SetVRS(bool enable);
	bool GetVRS();

	inline ContextType GetContextType() 
	{ 
		return ContextType::D3D11; 
	}
private:
	D3D11Context(HWND hWnd);

	std::vector<IDXGIAdapter*> QueryAdapters();
	IDXGIAdapter* GetLatestDiscreteAdapter();

	void CreateDeviceAndContext();
	void CreateSwapChain(
		HWND hWnd, UINT winWidth, UINT winHeight);


	void SetupViewport(UINT winWidth, UINT winHeight);
	void SetupDebugLayer();

	Microsoft::WRL::ComPtr<ID3D11Device> m_pDevice;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_pImmediateContext;
	Microsoft::WRL::ComPtr<IDXGISwapChain1> m_pSwapChain;

	D3D11_VIEWPORT m_viewport;

	//Debugging Tools
	Microsoft::WRL::ComPtr<ID3D11Debug> m_pDebugLayer;
	Microsoft::WRL::ComPtr<ID3D11InfoQueue> m_pInfoQueue;
	Microsoft::WRL::ComPtr<IDXGIInfoQueue> m_pDXGIInfoQueue;

	void InitializeNvAPI();
	void ShutdownNvAPI();
	NV_D3D1x_GRAPHICS_CAPS QueryGraphicsCapabilities();

	NV_D3D1x_GRAPHICS_CAPS m_gfxCaps;
	bool m_enableNvAPI;
	bool m_enableVRS;
};
