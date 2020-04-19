//DepthMap.hlsli

struct VS_INPUT
{
    float4 position : POSITION0;
};

struct GS_INPUT
{
    float4 position : SV_Position;
};

struct PS_INPUT
{
    float4 position : SV_Position;
};