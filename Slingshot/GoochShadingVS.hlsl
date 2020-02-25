#include "GoochShading.hlsli"

cbuffer VS_CONSTANT_BUFFER : register(b0)
{
    float4x4 wvpMatrix;
    
    float4 camPos;
    
    float4 lightPos;
    float4 lightColor;
    float lightInt;
};

PS_INPUT main(VS_INPUT vs_input)
{
    PS_INPUT vs_output;
    vs_output.position = mul(vs_input.position, wvpMatrix);
    vs_output.normal = vs_input.normal;
    
    float4 viewDir = float4(normalize(camPos.xyz - vs_input.position.xyz), 0.0f);
    float4 lightDir = float4(normalize(lightPos.xyz - vs_input.position.xyz), 0.0f);
    
    float4 coolColor = float4(0.0f, 0.0f, 0.55f, 0.0f) + float4(0.25f * vs_input.color.xyz, vs_input.color.w);
    float4 warmColor = float4(0.3f, 0.3f, 0.0f, 0.0f) + float4(0.25f * vs_input.color.xyz, vs_input.color.w);
    float4 highlightColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
    
    float t = (dot(vs_input.normal, lightDir) + 1.0f) / 2.0f;
    float4 r = 2.0f * dot(vs_input.normal, lightDir) * vs_input.normal - lightDir;
    float s = clamp(100.0f * dot(r, viewDir) - 97.0f, 0.0f, 1.0f);
    
    vs_output.color = s * highlightColor + (1.0f - s) * (t * warmColor + (1 - t) * coolColor);  
    return vs_output;
}