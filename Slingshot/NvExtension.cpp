#include "NvExtension.h"

NvExtension* NvExtension::Create()
{
	return new NvExtension();
}

NvExtension::NvExtension() :
	m_enabledVRS(false)
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
	return true;
}

void NvExtension::Shutdown()
{
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
			for (unsigned int i = 0; i < NV_MAX_PIXEL_SHADING_RATES; i++)
			{
				viewportShadingRateDesc.pViewports[0].shadingRateTable[i] = NV_PIXEL_SHADING_RATE::NV_PIXEL_X1_PER_4X4_RASTER_PIXELS;
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
	DXGI_SWAP_CHAIN_DESC1 swapChainDesc;
	DX::ThrowIfFailed(
		context.GetSwapChain()->GetDesc1(&swapChainDesc));

	ID3D11Texture2D* srs;
	D3D11_TEXTURE2D_DESC srsDesc;
	srsDesc.Width = swapChainDesc.Width / NV_VARIABLE_PIXEL_SHADING_TILE_WIDTH;
	srsDesc.Height = swapChainDesc.Height / NV_VARIABLE_PIXEL_SHADING_TILE_HEIGHT;
	srsDesc.ArraySize = 1;
	srsDesc.Format = DXGI_FORMAT_R8_UINT;
	srsDesc.SampleDesc.Count = 1;
	srsDesc.SampleDesc.Quality = 0;
	srsDesc.Usage = D3D11_USAGE_DEFAULT;
	srsDesc.CPUAccessFlags = 0;
	srsDesc.MiscFlags = 0;
	srsDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	srsDesc.MipLevels = 1;
	DX::ThrowIfFailed(
		context.GetDevice()->CreateTexture2D(&srsDesc, nullptr, &srs));

	NV_D3D11_SHADING_RATE_RESOURCE_VIEW_DESC srrvDesc;
	ZeroMemory(&srrvDesc, sizeof(NV_D3D11_SHADING_RATE_RESOURCE_VIEW_DESC));
	srrvDesc.version = NV_D3D11_SHADING_RATE_RESOURCE_VIEW_DESC_VER;
	srrvDesc.ViewDimension = NV_SRRV_DIMENSION_TEXTURE2D;
	srrvDesc.Texture2D.MipSlice = 0;
	srrvDesc.Format = DXGI_FORMAT_R8_UINT;
	ID3D11NvShadingRateResourceView* srsSRRV;
	NvAPI_Status NvStatus = NvAPI_D3D11_CreateShadingRateResourceView(context.GetDevice(), srs, &srrvDesc, &srsSRRV);
	if (NvStatus != NVAPI_OK)
	{
		return false;
	}

	SHADING_RATE_LOOKUP_TABLE srlt;
	srlt.shadingRate[0] = NV_PIXEL_X1_PER_RASTER_PIXEL;
	srlt.shadingRate[1] = NV_PIXEL_X1_PER_RASTER_PIXEL;
	srlt.shadingRate[2] = NV_PIXEL_X1_PER_RASTER_PIXEL;
	srlt.shadingRate[3] = NV_PIXEL_X1_PER_RASTER_PIXEL;
	srlt.shadingRate[4] = NV_PIXEL_X1_PER_RASTER_PIXEL;
	srlt.shadingRate[5] = NV_PIXEL_X1_PER_RASTER_PIXEL;
	srlt.shadingRate[6] = NV_PIXEL_X1_PER_2X2_RASTER_PIXELS;
	srlt.shadingRate[7] = NV_PIXEL_X1_PER_4X4_RASTER_PIXELS;
	srlt.shadingRate[8] = NV_PIXEL_X1_PER_4X4_RASTER_PIXELS;
	srlt.shadingRate[9] = NV_PIXEL_X1_PER_4X4_RASTER_PIXELS;
	srlt.shadingRate[10] = NV_PIXEL_X0_CULL_RASTER_PIXELS;
	srlt.shadingRate[11] = NV_PIXEL_X1_PER_RASTER_PIXEL;
	srlt.shadingRate[12] = NV_PIXEL_X1_PER_RASTER_PIXEL;
	srlt.shadingRate[13] = NV_PIXEL_X1_PER_RASTER_PIXEL;
	srlt.shadingRate[14] = NV_PIXEL_X1_PER_RASTER_PIXEL;
	srlt.shadingRate[15] = NV_PIXEL_X1_PER_RASTER_PIXEL;

	//Pending full implementation:
	//Populate Shading Rate Surface

	SAFE_RELEASE(srs);
	return false;
}
