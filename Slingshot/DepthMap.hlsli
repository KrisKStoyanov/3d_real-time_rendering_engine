//DepthMap.hlsli

struct VS_INPUT
{
    float4 position : POSITION0;
};

struct PS_INPUT
{
    float4 position : SV_Position;
    float4 depthPos : TEXCOORD0;
};