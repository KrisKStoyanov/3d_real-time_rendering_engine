//#pragma once
//#include "D3D11Context.h"
//
//#include "nvapi.h"
//
//class NvContextExtension
//{
//public:
//	NvContextExtension();
//	bool Initialize();
//	void Shutdown();
//
//	NV_D3D1x_GRAPHICS_CAPS QueryGraphicsCapabilities();
//
//	void PopulateShadingRateLookupTable();
//	NV_PIXEL_SHADING_RATE m_shadingRateLookupTable[NV_MAX_PIXEL_SHADING_RATES];
//
//	NV_D3D1x_GRAPHICS_CAPS m_gfxCaps;
//	bool m_enableNvAPI;
//	bool m_enableVRS;
//	ID3D11Texture2D* m_pShadingRateSurface;
//	ID3D11NvShadingRateResourceView* m_pShadingRateResourceView;
//
//	void PrintErrorNvAPI(NvAPI_Status status);
//};