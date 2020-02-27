struct VS_INPUT
{
    float4 position : POSITION;
    float4 color : COLOR;
    float4 normal : NORMAL;
};

struct PS_INPUT
{
    float4 position : SV_Position;
    float4 posWorld : W_POSITION;
    
    float4 normalWorld : W_NORMAL;
    
    float4 color : COLOR;
};

struct PS_OUTPUT
{
    float4 color : SV_Target;
};
