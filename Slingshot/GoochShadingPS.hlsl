#include "GoochShading.hlsli"

cbuffer PS_CONSTANT_BUFFER : register(b0)
{
    float4 camPos;
    
    float4 lightPos;
    float4 lightColor;
    float lightInt;
}

float4 unlit(float4 normal, float4 viewDir)
{
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
}

float4 lit(float4 lightDir, float4 normal, float4 viewDir)
{
    return float4(1.0f, 1.0f, 1.0f, 1.0f);
}

PS_OUTPUT main(PS_INPUT ps_input)
{
    PS_OUTPUT ps_output;
    float4 normal = normalize(ps_input.normalWorld);
    
    float4 viewDir = float4(normalize(camPos.xyz - ps_input.posWorld.xyz), 0.0f);
    float4 lightDir = float4(normalize(lightPos.xyz - ps_input.posWorld.xyz), 0.0f);
    
    float4 surfaceAmpColor = float4(0.25f * ps_input.color.xyz, ps_input.color.w);
    
    float4 coolColor = float4(0.0f, 0.0f, 0.55f, 0.0f) + surfaceAmpColor;
    float4 warmColor = float4(0.3f, 0.3f, 0.0f, 0.0f) + surfaceAmpColor;
    float4 highlightColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
     
    float t = (dot(normal, lightDir) + 1.0f) / 2.0f;
    float4 r = reflect(-lightDir, normal);
    float s = clamp(100.0f * dot(r, viewDir) - 97.0f, 0.0f, 1.0f);
    
     //vs_output.color = s * highlightColor + (1.0f - s) * (t * warmColor + (1 - t) * coolColor);  
    //vs_output.color = unlit(vs_input.normal, viewDir) + max(dot(lightDir, vs_input.normal), 0.0f) * lightColor * vs_input.color;
    //vs_output.color = 0.5f * coolColor + max(dpL, 0.0f) * lightColor * (s * highlightColor + (1.0f - s) * warmColor);
    
    //ps_output.color = 0.5f * coolColor + max(dot(lightDir, normal), 0.0f) * lightColor * (s * highlightColor + (1.0f - s) * warmColor);
    ps_output.color = 0.5f * coolColor + max(dot(normal, lightDir), 0.0f) * lightColor * (s * highlightColor + (1.0f - s) * warmColor);
    return ps_output;
}