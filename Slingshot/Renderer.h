#pragma once
#include "Stage.h"
#include "PipelineState.h"

struct RENDERER_DESC 
{
	GraphicsContextType graphicsContextType = GraphicsContextType::D3D11;
};

class Renderer {
public:
	static Renderer* Create(HWND hWnd, RENDERER_DESC& renderer_desc);
	bool Initialize();
	void OnFrameRender(Stage& stage);
	void Render(Camera& camera, Transform& transform, Mesh& mesh, PipelineState& props);
	void Shutdown();

	D3D11Context* GetGraphicsContext();
	PipelineState* GetPipelineState(VertexType vertexType); //pass enum with vertex type to get interpreting pipeline state
	void SetPipelineState(PIPELINE_DESC pipeline_desc, VertexType vertexType);
private:
	Renderer(HWND hWnd, RENDERER_DESC& renderer_desc);
	
	D3D11Context* m_pGraphicsContext;
	PipelineState* m_pPipelineState;
};