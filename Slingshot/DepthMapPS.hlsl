#include "DepthMap.hlsli"

cbuffer PerFrameBuffer : register(b0)
{
    float4x4 viewMatrix0;	
    float4x4 projectionMatrix;
}

float4 main(PS_INPUT ps_input) : SV_Target
{
    float depthValue = ps_input.depthPos.z / ps_input.depthPos.w;
    float4 color = float4(depthValue, depthValue, depthValue, 1.0f);
	return color;
}