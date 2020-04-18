#pragma once
#include "D3D11Context.h"
#include "nvapi.h"

struct SHADING_RATE_LOOKUP_TABLE
{
	NV_PIXEL_SHADING_RATE shadingRate[NV_MAX_PIXEL_SHADING_RATES];
};

class NvExtension
{
public:
	static NvExtension* Create();
	bool Initialize(D3D11Context& context);
	void Shutdown();
	bool SetConstantVRS(bool enabled, D3D11Context& context);
	bool SetVRSwithSRS(bool enabled, D3D11Context& context);

	inline bool GetVRS()
	{
		return m_enabledVRS;
	}
private:
	NvExtension();
	bool m_enabledVRS;
	NV_D3D1x_GRAPHICS_CAPS m_gfxCaps;
	SHADING_RATE_LOOKUP_TABLE m_srlt;
	ID3D11Texture2D* m_pSrs;
	ID3D11NvShadingRateResourceView* m_pSrsSRRV;
	int m_constShadingRateIndex;
};

