#include "NvExtension.h"

NvExtension* NvExtension::Create() 
{
	return new NvExtension();
}

NvExtension::NvExtension() :
	m_enabledVRS(false),
	m_constShadingRateIndex(0)
{

}

bool NvExtension::Initialize(D3D11Context& context)
{
	NvAPI_Status NvStatus = NvAPI_Initialize();
	if (NvStatus != NVAPI_OK)
	{
		return false;
	}
	memset(&m_gfxCaps, 0, sizeof(NV_D3D1x_GRAPHICS_CAPS));
	NvStatus = NvAPI_D3D1x_GetGraphicsCapabilities(context.GetDevice(), NV_D3D1x_GRAPHICS_CAPS_VER, &m_gfxCaps);
	if (NvStatus != NVAPI_OK)
	{
		return false;
	}

	if (m_gfxCaps.bVariablePixelRateShadingSupported)
	{
		m_srlt.shadingRate[0] = NV_PIXEL_X1_PER_RASTER_PIXEL;
		m_srlt.shadingRate[1] = NV_PIXEL_X1_PER_1X2_RASTER_PIXELS;
		m_srlt.shadingRate[2] = NV_PIXEL_X1_PER_2X1_RASTER_PIXELS;
		m_srlt.shadingRate[3] = NV_PIXEL_X1_PER_2X2_RASTER_PIXELS;
		m_srlt.shadingRate[4] = NV_PIXEL_X1_PER_2X4_RASTER_PIXELS;
		m_srlt.shadingRate[5] = NV_PIXEL_X1_PER_4X2_RASTER_PIXELS;
		m_srlt.shadingRate[6] = NV_PIXEL_X1_PER_4X4_RASTER_PIXELS;
		m_srlt.shadingRate[7] = NV_PIXEL_X1_PER_RASTER_PIXEL;
		m_srlt.shadingRate[8] = NV_PIXEL_X1_PER_1X2_RASTER_PIXELS;
		m_srlt.shadingRate[9] = NV_PIXEL_X1_PER_2X1_RASTER_PIXELS;
		m_srlt.shadingRate[10] = NV_PIXEL_X1_PER_2X2_RASTER_PIXELS;
		m_srlt.shadingRate[11] = NV_PIXEL_X1_PER_2X4_RASTER_PIXELS;
		m_srlt.shadingRate[12] = NV_PIXEL_X1_PER_4X2_RASTER_PIXELS;
		m_srlt.shadingRate[13] = NV_PIXEL_X1_PER_4X4_RASTER_PIXELS;
		m_srlt.shadingRate[14] = NV_PIXEL_X1_PER_4X4_RASTER_PIXELS;
		m_srlt.shadingRate[15] = NV_PIXEL_X1_PER_4X4_RASTER_PIXELS;
	}

	return true;
}

void NvExtension::Shutdown()
{
	SAFE_RELEASE(m_pSrs);
	SAFE_RELEASE(m_pSrsSRRV);
	NvAPI_Unload();
}

bool NvExtension::SetConstantVRS(bool enabled, D3D11Context& context)
{
	if (!m_gfxCaps.bVariablePixelRateShadingSupported)
	{
		return false;
	}
	NV_D3D11_VIEWPORTS_SHADING_RATE_DESC viewportShadingRateDesc;
	ZeroMemory(&viewportShadingRateDesc, sizeof(NV_D3D11_VIEWPORTS_SHADING_RATE_DESC));
	viewportShadingRateDesc.version = NV_D3D11_VIEWPORTS_SHADING_RATE_DESC_VER;
	if (enabled)
	{
		viewportShadingRateDesc.numViewports = 1;
		viewportShadingRateDesc.pViewports =
			(NV_D3D11_VIEWPORT_SHADING_RATE_DESC*)malloc(sizeof(NV_D3D11_VIEWPORT_SHADING_RATE_DESC));
		if (viewportShadingRateDesc.pViewports)
		{
			viewportShadingRateDesc.pViewports[0].enableVariablePixelShadingRate = true;
			//Cycle the rate if already enabled
			if (m_enabledVRS)
			{
				m_constShadingRateIndex++;
				if (m_constShadingRateIndex > 6)
				{
					m_constShadingRateIndex = 0;
				}
			}
			for (unsigned int i = 0; i < NV_MAX_PIXEL_SHADING_RATES; i++)
			{
				viewportShadingRateDesc.pViewports[0].shadingRateTable[i] = m_srlt.shadingRate[m_constShadingRateIndex];
			}
			NvAPI_Status NvStatus = NvAPI_D3D11_RSSetViewportsPixelShadingRates(context.GetContext(), &viewportShadingRateDesc);
			if (NvStatus != NVAPI_OK)
			{
				return false;
			}
			NvStatus = NvAPI_D3D11_RSSetShadingRateResourceView(context.GetContext(), nullptr);
			if (NvStatus != NVAPI_OK)
			{
				return false;
			}
		}
	}
	else
	{
		viewportShadingRateDesc.numViewports = 0;
		NvAPI_Status NvStatus = NvAPI_D3D11_RSSetViewportsPixelShadingRates(context.GetContext(), &viewportShadingRateDesc);
		if (NvStatus != NVAPI_OK)
		{
			return false;
		}
	}

	m_enabledVRS = enabled;
	return true;
}

bool NvExtension::SetVRSwithSRS(bool enabled, D3D11Context& context)
{
	if (!m_gfxCaps.bVariablePixelRateShadingSupported)
	{
		return false;
	}

	////Disable viewport shading rate settings
	//NV_D3D11_VIEWPORTS_SHADING_RATE_DESC viewportShadingRateDesc;
	//ZeroMemory(&viewportShadingRateDesc, sizeof(NV_D3D11_VIEWPORTS_SHADING_RATE_DESC));
	//viewportShadingRateDesc.version = NV_D3D11_VIEWPORTS_SHADING_RATE_DESC_VER;
	//viewportShadingRateDesc.numViewports = 0;
	//NvAPI_Status NvStatus = NvAPI_D3D11_RSSetViewportsPixelShadingRates(context.GetContext(), &viewportShadingRateDesc);
	//if (NvStatus != NVAPI_OK)
	//{
	//	return false;
	//}

	if (enabled)
	{
		NvAPI_Status NvStatus = NvAPI_D3D11_RSSetShadingRateResourceView(context.GetContext(), m_pSrsSRRV);
		if (NvStatus != NVAPI_OK)
		{
			return false;
		}

	}
	else 
	{
		NvAPI_Status NvStatus = NvAPI_D3D11_RSSetShadingRateResourceView(context.GetContext(), nullptr);
		if (NvStatus != NVAPI_OK)
		{
			return false;
		}
	}
	
	m_enabledVRS = enabled;
	return true;
}
