struct VS_INPUT
{
    float4 position : POSITION0;
    float4 normal : NORMAL0;
};

struct PS_INPUT
{
    float4 position : SV_Position;
    float4 posWorld : W_POSITION;
    
    float4 normalWorld : W_NORMAL;
};

struct PS_OUTPUT
{
    float4 color : SV_Target;
};
