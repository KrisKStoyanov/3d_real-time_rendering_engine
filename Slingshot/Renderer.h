#pragma once
#include "Model.h"
#include "Camera.h"

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
	void OnFrameRender(Model* model, Transform* transform, Camera* camera);
	void Shutdown();

	D3D11Context* GetGraphicsContext();
private:
	Renderer(HWND hWnd, RENDERER_DESC* renderer_desc);
	
	D3D11Context* m_pGraphicsContext;
};