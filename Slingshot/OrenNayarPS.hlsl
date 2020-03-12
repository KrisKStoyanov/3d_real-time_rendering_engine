#include "OrenNayar.hlsli"

cbuffer PS_CONSTANT_BUFFER : register(b0)
{
    float4 camPos;
    
    float4 lightPos;
    float4 lightColor;
    
    float4 surfaceColor;
    
    float roughness;
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
    
    float4 p = surfaceColor;
    float lightDirAngle = dot(lightDir, normal) / (length(lightDir) * length(normal));
    float4 radianceReflL = dot(dot(p / 3.14f, lightDirAngle), 1.f);
    
    ps_output.color = surfaceColor * radianceReflL;
    return ps_output;
}