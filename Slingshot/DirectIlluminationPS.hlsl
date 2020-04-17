//GoochIlluminationPS.hlsl
#include "GoochIllumination.hlsli"

Texture2D depthMapTexture : register(t0);

SamplerState sampleTypeClamp : register(s0);

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
    float4 color = ambientColor;
    float4 normal = normalize(ps_input.normalWorld);
    
    float2 projectTexCoord;
    projectTexCoord.x = ps_input.lightViewPos.x / ps_input.lightViewPos.w / 2.0f + 0.5f;
    projectTexCoord.y = -ps_input.lightViewPos.y / ps_input.lightViewPos.w / 2.0f + 0.5f;
    
    if ((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y))
    {
        float depthValue = depthMapTexture.Sample(sampleTypeClamp, projectTexCoord).r;
        float lightDepthValue = (ps_input.lightViewPos.z / ps_input.lightViewPos.w) - bias;
        
        if (lightDepthValue < depthValue)
        {
            float lightIntensity = saturate(dot(normal, ps_input.lightDir));
            if (lightIntensity > 0.0f)
            {
                color += (diffuseColor * lightIntensity);
                color = saturate(color);
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

    return color * blendColor;
}