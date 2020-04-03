#include "FinalGathering.hlsli"

cbuffer PerFrameBuffer : register(b0)
{
    float4 camPos;
    float4 lightPos;
    float4 lightColor;
}

cbuffer PerDrawCallBuffer : register(b1)
{
    float4 surfaceColor;
}

PS_OUTPUT main(PS_INPUT ps_input)
{
    PS_OUTPUT ps_output;
    
    ps_output.color = surfaceColor;
    return ps_output;
}