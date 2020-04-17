#pragma once
#include "D3D11Context.h"
#include "nvapi.h"

struct SHADING_RATE_LOOKUP_TABLE
{
	NV_PIXEL_SHADING_RATE shadingRate[16];
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
};

