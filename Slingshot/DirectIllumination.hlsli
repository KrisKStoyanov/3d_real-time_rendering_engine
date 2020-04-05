struct VS_INPUT
{
    float4 position : POSITION0;
    float2 uv : TEXCOORD0;
    float4 normal : NORMAL0;
};

struct PS_INPUT
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 normal : NORMAL;
    float4 posWorld : W_POSITION;
};
