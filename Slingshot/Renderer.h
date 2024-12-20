#pragma once
#include "Scene.h"
#include "D3D11DepthMap.h"
#include "D3D11DirectIllumination.h"
#include "NvExtension.h"
#include "PhotonMap.h"

struct RENDERER_DESC 
{
	ContextType gfxContextType = ContextType::D3D11;
};

class Renderer {
public:
	static Renderer* Create(HWND hWnd, RENDERER_DESC& renderer_desc);
	bool Initialize();
	void Draw(Scene& scene);
	void Shutdown();
	void UpdateConstantVRS();
	void ToggleConstantVRS();
	void ToggleVRSwithSRS();

	GraphicsContext* GetGraphicsContext();

	void SetupPhotonMap(GEOMETRY_DESC& desc);

private:
	Renderer(HWND hWnd, RENDERER_DESC& renderer_desc);	
	D3D11Context* m_pGraphicsContext;

	D3D11DepthMap* m_pDepthMap;
	D3D11DirectIllumination* m_pDirectIllumination;

	NvExtension* m_pNvExtension;

	PhotonMap* m_pPhotonMap;
};