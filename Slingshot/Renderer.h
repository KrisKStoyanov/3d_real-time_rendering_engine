#pragma once
//Include other supported contexts for multiple setup & render variants
#include "Camera.h"
#include "Entity.h"

class Renderer {
public:
	bool Initialize(HWND hWnd, GraphicsContextType graphicsContextType);
	void OnFrameRender();
	void Shutdown();

	bool Setup(D3D11Context* context, Entity* entity);
	void Render(D3D11Context* context, Entity* entity);
private:
	D3D11Context* m_pGraphicsContext;
};