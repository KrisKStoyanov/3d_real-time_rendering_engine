//GoochIlluminationPS.hlsl
#include "GoochIllumination.hlsli"

Texture2D shadowMap : register(t0);
SamplerState samplerShadowMap : register(s0);

cbuffer PerFrameBuffer : register(b0)
{
    float4 camPos;
    float4 lightPos;
    float4 ambientColor;
    float4 diffuseColor;
}

cbuffer PerDrawCallBuffer : register(b1)
{
    float4 surfaceColor;
}

float4 main(PS_INPUT ps_input) : SV_Target
{
    float bias = 0.01f;
    float4 lightColor = ambientColor;
    float4 normal = normalize(ps_input.normalWorld);
    
    float2 projectTexCoord;
    projectTexCoord.x = 0.5f + (ps_input.posLightWorld.x / ps_input.posLightWorld.w * 0.5f);
    projectTexCoord.y = 0.5f - (ps_input.posLightWorld.y / ps_input.posLightWorld.w * 0.5f);
    float lightDepthValue = (ps_input.posLightWorld.z / ps_input.posLightWorld.w) - bias;
    
    if ((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y) && (lightDepthValue > 0.0f))
    {
        float depthValue = shadowMap.Sample(samplerShadowMap, projectTexCoord).r;
        
        if (lightDepthValue < depthValue)
        {
            float lightIntensity = saturate(dot(normal, ps_input.lightRay));
            if (lightIntensity > 0.0f)
            {
                lightColor += (diffuseColor * lightIntensity);
                lightColor = saturate(lightColor);
            }
        }
    }
    
    float4 viewDir = float4(normalize(camPos.xyz - ps_input.posWorld.xyz), 0.0f);
    float4 lightDir = float4(normalize(lightPos.xyz - ps_input.posWorld.xyz), 0.0f);
    
    float4 surfaceAmpColor = float4(0.25f * surfaceColor.xyz, surfaceColor.w);
    
    float4 coolColor = float4(0.0f, 0.0f, 0.55f, 0.0f) + surfaceAmpColor;
    float4 warmColor = float4(0.3f, 0.3f, 0.0f, 0.0f) + surfaceAmpColor;
    float4 highlightColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
     
    float t = (dot(normal, lightDir) + 1.0f) / 2.0f;
    float4 r = reflect(-lightDir, normal);
    float s = clamp(100.0f * dot(r, viewDir) - 97.0f, 0.0f, 1.0f);
    
    float4 blendColor = 0.5f * coolColor + max(dot(normal, lightDir), 0.0f) * diffuseColor * (s * highlightColor + (1.0f - s) * warmColor);

    return lightColor * blendColor;
}