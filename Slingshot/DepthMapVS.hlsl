//DepthMapVS.hlsl
#include "DepthMap.hlsli"

cbuffer PerDrawCallBuffer : register(b0)
{
    float4x4 worldMatrix;
}

GS_COALESCENT main(VS_INPUT vs_input)
{
    GS_COALESCENT vs_output;
 
    vs_output.position = mul(vs_input.position, worldMatrix);
    vs_output.depthPos = vs_output.position;
    
    return vs_output;
}