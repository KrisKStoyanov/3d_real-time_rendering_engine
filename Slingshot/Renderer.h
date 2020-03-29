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
	bool Initialize(PIPELINE_DESC pipeline_desc);
	void Draw(Scene& scene);
	void Shutdown();

	D3D11Context* GetGraphicsContext();
private:
	Renderer(HWND hWnd, RENDERER_DESC& renderer_desc);	
	D3D11Context* m_pGraphicsContext;
};