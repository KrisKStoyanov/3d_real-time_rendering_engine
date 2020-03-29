#pragma once
#include "Scene.h"
#include "D3D11PipelineState.h"

struct RENDERER_DESC 
{
	gfx::ContextType gfxContextType = gfx::ContextType::D3D11;
};

class Renderer {
public:
	static Renderer* Create(HWND hWnd, RENDERER_DESC& renderer_desc);
	bool Initialize();
	void Draw(Scene& scene);
	void Shutdown();

	D3D11Context* GetGraphicsContext();
	D3D11PipelineState* GetPipelineState(ShadingModel shadingModel); //pass enum with vertex type to get interpreting pipeline state
	void SetPipelineState(const PIPELINE_DESC& pipelineDesc);
private:
	Renderer(HWND hWnd, RENDERER_DESC& renderer_desc);
	
	D3D11Context* m_pGraphicsContext;
	D3D11PipelineState* m_pPipelineState;
};