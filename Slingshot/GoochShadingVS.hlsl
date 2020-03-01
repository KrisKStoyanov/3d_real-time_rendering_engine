#include "GoochShading.hlsli"

cbuffer VS_CONSTANT_BUFFER : register(b0)
{
    float4x4 worldMatrix;
    float4x4 viewMatrix;
    float4x4 projMatrix;
};

PS_INPUT main(VS_INPUT vs_input)
{
    PS_INPUT vs_output;
 
    float4x4 wvpMatrix = mul(worldMatrix, mul(viewMatrix, projMatrix));
     
    vs_output.position = mul(vs_input.position, wvpMatrix);
    
    vs_output.posWorld = mul(vs_input.position, worldMatrix);   
    vs_output.normalWorld = mul(vs_input.normal, worldMatrix);
    
    return vs_output;
}