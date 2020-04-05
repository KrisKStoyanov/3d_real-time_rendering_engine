#include "FinalGathering.hlsli"
Texture2D shaderTexture : register(t0);
Texture2D depthMapTexture : register(t1);

SamplerState sampleTypeClamp : register(s0);
SamplerState sampleTypeWrap : register(s1);

cbuffer PerFrameBuffer : register(b0)
{
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
    float2 projectTexCoord;
    float depthValue;
    float lightDepthValue;
    float lightIntensity;
    float4 textureColor;
    
    float4 normal = normalize(ps_input.normalWorld);
    
    projectTexCoord.x = ps_input.lightViewPos.x / ps_input.lightViewPos.w / 2.0f + 0.5f;
    projectTexCoord.y = -ps_input.lightViewPos.y / ps_input.lightViewPos.w / 2.0f + 0.5f;
    
    if (saturate(projectTexCoord.x) == projectTexCoord.x && saturate(projectTexCoord.y) == projectTexCoord.y)
    {
        depthValue = depthMapTexture.Sample(sampleTypeClamp, projectTexCoord).r;
        lightDepthValue = (ps_input.lightViewPos.z / ps_input.lightViewPos.w) - bias;
        
        if (lightDepthValue < depthValue)
        {
            lightIntensity = saturate(dot(normal, ps_input.lightDir));
            if (lightIntensity > 0.0f)
            {
                color += (diffuseColor * lightIntensity);
                color = saturate(color);
            }
        }
    }
    textureColor = shaderTexture.Sample(sampleTypeWrap, ps_input.uv);
    color *= textureColor;

    return surfaceColor;
}