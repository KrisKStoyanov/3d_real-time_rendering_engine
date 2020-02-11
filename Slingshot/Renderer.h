#pragma once
#include "Camera.h"
#include "Entity.h"

struct RENDERER_DESC {
	GraphicsContextType graphicsContextType;
	RENDERER_DESC(
		GraphicsContextType _graphicsContextType) : 
		graphicsContextType(_graphicsContextType) {}
};

class Renderer {
public:
	static Renderer* Create(HWND hWnd, RENDERER_DESC* renderer_desc);
	bool Initialize();
	void OnFrameRender();
	void Shutdown();

	bool Setup(D3D11Context* context, Entity* entity);
	void Render(D3D11Context* context, Entity* entity);
private:
	Renderer(HWND hWnd, RENDERER_DESC* renderer_desc);
	RENDERER_DESC* m_pDesc;
	
	D3D11Context* m_pGraphicsContext;
};