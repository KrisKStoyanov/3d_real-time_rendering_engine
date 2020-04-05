#pragma once
#include "Scene.h"
#include "D3D11Context.h"
#include "PhotonMap.h"

struct RENDERER_DESC 
{
	ContextType gfxContextType = ContextType::D3D11;
};

class Renderer {
public:
	static Renderer* Create(HWND hWnd, RENDERER_DESC& renderer_desc);
	bool Initialize(PIPELINE_DESC pipeline_desc);
	void Draw(Scene& scene);
	void Shutdown();

	GraphicsContext* GetGraphicsContext();

	void SetupPhotonMap(GEOMETRY_DESC& desc);
private:
	Renderer(HWND hWnd, RENDERER_DESC& renderer_desc);	
	D3D11Context* m_pGraphicsContext;
	D3D11PipelineState* m_pPipelineState;

	PhotonMap* m_pPhotonMap;
};