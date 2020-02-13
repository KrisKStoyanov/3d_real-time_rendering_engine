#include "ColorShader.hlsli"

cbuffer VS_CONSTANT_BUFFER : register(b0)
{
    float4x4 wvpMatrix;
};

PS_INPUT main(VS_INPUT vs_input)
{
    PS_INPUT vs_output;
    vs_output.position = mul(vs_input.position, wvpMatrix);
    vs_output.color = vs_input.color;
    return vs_output;
}