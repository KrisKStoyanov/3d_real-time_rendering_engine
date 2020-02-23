#pragma once
#include "Stage.h"

struct RENDERER_DESC 
{
	GraphicsContextType graphicsContextType = GraphicsContextType::D3D11;
};

class Renderer {
public:
	static Renderer* Create(HWND hWnd, RENDERER_DESC& renderer_desc);
	bool Initialize();
	void OnFrameRender(Stage& stage);
	void Shutdown();

	D3D11Context* GetGraphicsContext();
private:
	Renderer(HWND hWnd, RENDERER_DESC& renderer_desc);
	
	D3D11Context* m_pGraphicsContext;
};