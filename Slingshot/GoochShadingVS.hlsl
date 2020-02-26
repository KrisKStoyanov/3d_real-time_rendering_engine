#include "GoochShading.hlsli"

cbuffer VS_CONSTANT_BUFFER : register(b0)
{
    float4x4 wvpMatrix;
    
    float4 camPos;
    
    float4 lightPos;
    float4 lightColor;
    float lightInt;
};

float4 unlit(float4 normal, float4 viewDir)
{
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
}

float4 lit(float4 lightDir, float4 normal, float4 viewDir)
{
    return float4(1.0f, 1.0f, 1.0f, 1.0f);
}

PS_INPUT main(VS_INPUT vs_input)
{
    PS_INPUT vs_output;
    vs_output.position = mul(vs_input.position, wvpMatrix);
    vs_output.normal = vs_input.normal;
    
    float4 viewDir = float4(normalize(camPos.xyz - vs_input.position.xyz), 0.0f);
    float4 lightDir = float4(normalize(lightPos.xyz - vs_input.position.xyz), 0.0f);
    
    float4 surfaceAmpColor = float4(0.25f * vs_input.color.xyz, vs_input.color.w);
    
    float4 coolColor = float4(0.0f, 0.0f, 0.55f, 0.0f) + surfaceAmpColor;
    float4 warmColor = float4(0.3f, 0.3f, 0.0f, 0.0f) + surfaceAmpColor;
    float4 highlightColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
    
    float dpL = dot(vs_input.normal, lightDir);
    
    float t = (dpL + 1.0f) / 2.0f;
    float4 r = 2.0f * dpL * vs_input.normal - lightDir;
    float s = clamp(100.0f * dot(r, viewDir) - 97.0f, 0.0f, 1.0f);
    
    //vs_output.color = s * highlightColor + (1.0f - s) * (t * warmColor + (1 - t) * coolColor);  
    vs_output.color = unlit(vs_input.normal, viewDir) + max(dot(lightDir, vs_input.normal), 0.0f) * lightColor * vs_input.color;
    return vs_output;
}