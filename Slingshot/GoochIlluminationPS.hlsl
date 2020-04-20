//GoochIlluminationPS.hlsl
#include "GoochIllumination.hlsli"

TextureCube<float> shadowMap : register(t0);
SamplerComparisonState samplerShadowMap : register(s0);

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

float vecToDepth(float4 vec, float n, float f)
{
    float4 absVec = abs(vec);
    float localZComp = max(absVec.x, max(absVec.y, absVec.z));
    float normZComp = (f + n) / (f - n) - (2.0f * f * n) / (f - n) / localZComp;
    return (normZComp + 1.0f) * 0.5f;
}

float4 main(PS_INPUT ps_input) : SV_Target
{
    float bias = 0.01f;
    float4 lightColor = ambientColor;
    float4 normal = normalize(ps_input.normalWorld);

    float4 lightRay = ps_input.posWorld - lightPos;
    float4 lightRayDir = normalize(lightRay);
    float sampledDepth = vecToDepth(lightRay, 1.0f, 1000.0f);
    
    lightColor = shadowMap.SampleCmpLevelZero(samplerShadowMap, float3(lightRayDir.x, lightRayDir.y, -lightRayDir.z), sampledDepth).r;
    //float2 projectTexCoord;
    //projectTexCoord.x = 0.5f + (ps_input.posLightWorld.x / ps_input.posLightWorld.w * 0.5f);
    //projectTexCoord.y = 0.5f - (ps_input.posLightWorld.y / ps_input.posLightWorld.w * 0.5f);
    //float lightDepthValue = (ps_input.posLightWorld.z / ps_input.posLightWorld.w) - bias;
    //if ((saturate(projectTexCoord.x) == projectTexCoord.x) && (saturate(projectTexCoord.y) == projectTexCoord.y) && (lightDepthValue > 0.0f))
    //{
        //float margin = acos(saturate(ps_input.lightRay));
        //float epsilon = 0.0005f / margin;
        //epsilon = clamp(epsilon, 0.0f, 0.1f);
        
        //float depthValue = shadowMap.SampleCmpLevelZero(samplerShadowMap, projectTexCoord, (lightDepthValue + epsilon)).r;
        
        //if (lightDepthValue < depthValue)
        //{
        //    float lightIntensity = saturate(dot(normal, ps_input.lightRay));
        //    if (lightIntensity > 0.0f)
        //    {
        //        lightColor += (diffuseColor * lightIntensity);
        //        lightColor = saturate(lightColor);
        //    }
        //}
    //}
    
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
    
    //oren-nayar
    //half VDotN = dot(viewDir, normal);
    //half LDotN = dot(lightDir, normal);
    //half cosThetaI = LDotN;
    //half thetaR = acos(VDotN);
    //half thetaI = acos(cosThetaI);
    //half cosPhiDiff = dot(normalize(viewDir - normal * VDotN), normalize(lightDir - normal * LDotN));
    //half alpha = max(thetaI, thetaR);
    //half beta = min(thetaI, thetaR);
    //half sigma2 = 1.0f * 1.0f;
    //half A = 1.0f - 0.5f * sigma2 / (sigma2 * 0.33f);
    //half B = 0.45f * sigma2 / (sigma2 + 0.09f);
    //return (saturate(cosThetaI) * (A + (B * saturate(cosPhiDiff) * sin(alpha) * tan(beta)))) * blendColor;
    
    //journey diffuse shader
    //normal = float4(normal.x, normal.y * 0.3f, normal.z, normal.w);
    //return saturate(4.0f * dot(normal, lightDir)) * blendColor;
}