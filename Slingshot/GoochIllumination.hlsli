//GoochIllumination.hlsli

struct VS_INPUT
{
    float4 position : POSITION0;
    float4 normal : NORMAL0;
    float2 uv : TEXCOORD0;
};

struct PS_INPUT
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 posWorld : W_POSITION;
    float4 posLightWorld : LW_POSITION;
    float4 normalWorld : NORMAL0;
    float4 lightRay : NORMAL1;
    float4 incRay : NORMAL2;
};
