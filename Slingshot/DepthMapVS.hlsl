//DepthMapVS.hlsl
#include "DepthMap.hlsli"

cbuffer PerDrawCallBuffer : register(b1)
{
    float4x4 worldMatrix;
}

PS_INPUT main(VS_INPUT vs_input)
{
    PS_INPUT vs_output = (PS_INPUT)0;
    vs_output.position = mul(vs_input.position, worldMatrix);

    return vs_output;
}