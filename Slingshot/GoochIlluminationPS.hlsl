//GoochIlluminationPS.hlsl
#include "GoochIllumination.hlsli"

Texture2D shadowMap : register(t0);
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

float vecToDepth(float4 vec, float nearZ, float farZ)
{
    float4 absVec = abs(vec);
    float localZComp = max(absVec.x, max(absVec.y, absVec.z));
    float normZComp = (farZ + nearZ) / (farZ - nearZ) - (2.0f * farZ * nearZ) / (farZ - nearZ) / localZComp;
    return (normZComp + 1.0f) * 0.5f;
}

float2 texelOffset(float u, float v)
{
    return float2(u * 1.0f / 512.0f, v * 1.0f / 512.0f);
}

float4 main(PS_INPUT ps_input) : SV_Target
{
    float4 normal = normalize(ps_input.normalWorld);
    float4 lightDir = float4(normalize(lightPos.xyz - ps_input.posWorld.xyz), 0.0f);
    float4 viewDir = float4(normalize(camPos.xyz - ps_input.posWorld.xyz), 0.0f);
    float4 surfaceAmpColor = float4(0.25f * surfaceColor.xyz, surfaceColor.w);
    float4 coolColor = float4(0.0f, 0.0f, 0.55f, 0.0f) + surfaceAmpColor;
    float4 warmColor = float4(0.3f, 0.3f, 0.0f, 0.0f) + surfaceAmpColor;
    float4 highlightColor = float4(1.0f, 1.0f, 1.0f, 1.0f);
     
    float t = (dot(normal, lightDir) + 1.0f) / 2.0f;
    float4 r = reflect(-lightDir, normal);
    float s = clamp(100.0f * dot(r, viewDir) - 97.0f, 0.0f, 1.0f);
    
    float4 blendColor = 0.5f * coolColor + max(dot(normal, lightDir), 0.0f) * diffuseColor * (s * highlightColor + (1.0f - s) * warmColor);
    
    float3 shadowTexCoord = ps_input.posLightWorld.xyz / ps_input.posLightWorld.w;
    shadowTexCoord.x = 0.5f + (shadowTexCoord.x * 0.5f);
    shadowTexCoord.y = 0.5f - (shadowTexCoord.y * 0.5f);
    float depthSample = shadowMap.SampleCmpLevelZero(samplerShadowMap, shadowTexCoord.xy, shadowTexCoord.z).r;
    float lightIntensity = saturate(dot(normal, lightDir));
    float4 lightColor = ambientColor * depthSample + diffuseColor * lightIntensity;
    lightColor = saturate(lightColor);
     
    return lightColor * blendColor;
    
    //float bias = 0.01f;
    //float margin = acos(saturate(lightDir));
    //float epsilon = 0.001f / margin;
    //epsilon = clamp(epsilon, 0.0f, 0.1f);
    //float lightIntensity = saturate(dot(normal, lightDir));
    //lightColor += (diffuseColor * lightIntensity);
    //lightColor = saturate(lightColor);
    
    //if (saturate(shadowTexCoord.x) == shadowTexCoord.x &&
    //    saturate(shadowTexCoord.y) == shadowTexCoord.y &&
    //    shadowTexCoord.z > 0.0f)
    //{
    //    float lightIntensity = saturate(dot(normal, lightDir));
    //    lightColor += diffuseColor * lightIntensity;
    //    lightColor = saturate(lightColor);
    //}
    
    //if ((saturate(shadowProjCoord.x) == shadowProjCoord.x) &&
    //    (saturate(shadowProjCoord.y) == shadowProjCoord.y) &&
    //    (saturate(shadowProjCoord.z) == shadowProjCoord.z) && (shadowProjCoord.z > 0.0f))
    //{
    //    float lightIntensity = saturate(dot(normal, lightDir));
    //    lightColor += (diffuseColor * lightIntensity) * shadowProjCoord.z;
    //    lightColor = saturate(lightColor);
    //}
    
    //float sum = 0;
    //float x, y;
 
    //    //4x4 Bilinear HPCF
    //for (y = -1.5f; y <= 1.5f; y += 1.0f);
    //    {
    //    for (x = -1.5f; x <= 1.5f; x += 1.0f)
    //    {
    //        sum += shadowMap.SampleCmpLevelZero(samplerShadowMap, ps_input.posLightWorld.xy + texelOffset(x, y), ps_input.posLightWorld.z);
    //    }
    //}
    //lightColor = (sum / 16.0f) * ambientColor;

    //---------
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