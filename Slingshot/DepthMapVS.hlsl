//DepthMapVS.hlsl
#include "DepthMap.hlsli"

cbuffer PerFrameBuffer : register(b0)
{
    float4x4 viewMatrix;
    float4x4 projMatrix;
};

cbuffer PerDrawCallBuffer : register(b1)
{
    float4x4 worldMatrix;
}

PS_INPUT main(VS_INPUT vs_input)
{
    PS_INPUT vs_output = (PS_INPUT)0;
    float4x4 wvpMatrix = mul(worldMatrix, mul(viewMatrix, projMatrix));
    vs_output.position = mul(vs_input.position, wvpMatrix);

    return vs_output;
}