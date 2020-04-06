#include "FinalGathering.hlsli"
Texture2D depthMapTexture : register(t0);

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
    //float4 textureColor = shaderTexture.Sample(sampleTypeWrap, ps_input.uv);
    //color *= textureColor;

    return surfaceColor;
}