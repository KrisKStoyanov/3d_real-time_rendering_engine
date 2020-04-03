#include "OrenNayar.hlsli"

cbuffer PerFrameBuffer : register(b0)
{
    float4 camPos;
    float4 lightPos;
    float4 lightColor;
}

cbuffer PerDrawCallBuffer : register(b1)
{
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
    
    //float4 p = surfaceColor;
    //float lightDirAngle = dot(lightDir, normal) / (length(lightDir) * length(normal));
    //float4 radianceReflL = dot(dot(p / 3.14f, lightDirAngle), roughness);

    float pi = 3.14f;
    
    float NdotL = dot(normal, lightDir);
    float NdotV = dot(normal, viewDir);

    float angleVN = acos(NdotV);
    float angleLN = acos(NdotL);

    float alpha = max(angleVN, angleLN);
    float beta = min(angleVN, angleLN);
    float gamma = dot(viewDir - normal * dot(viewDir, normal), lightDir - normal * dot(lightDir, normal));

    float roughnessSquared = roughness * roughness;
    float roughnessSquared9 = (roughnessSquared / (roughnessSquared + 0.09));

    // calculate C1, C2 and C3
    float C1 = 1.0 - 0.5 * (roughnessSquared / (roughnessSquared + 0.33));
    float C2 = 0.45 * roughnessSquared9;

    if (gamma > 0.0f)
    {
        C2 *= sin(alpha);
    }
    else
    {
        C2 *= (sin(alpha) - pow((2.0f * beta) / pi, 3.0f));
    }

    float powValue = (4.0f * alpha * beta) / (pi * pi);
    float C3 = 0.125f * roughnessSquared9 * powValue * powValue;

    // now calculate both main parts of the formula
    float A = gamma * C2 * tan(beta);
    float B = (1.0f - abs(gamma)) * C3 * tan((alpha + beta) / 2.0f);

    // put it all together
    float L1 = max(0.0f, NdotL) * (C1 + A + B);

    // also calculate interreflection
    float twoBetaPi = 2.0f * beta / pi;

    float L2 = 0.17f * max(0.0f, NdotL) * (roughnessSquared / (roughnessSquared + 0.13f)) * (1.0f - gamma * twoBetaPi * twoBetaPi);
    
    ps_output.color = float4(surfaceColor.xyz * (L1 + L2), surfaceColor.w);
    return ps_output;
}