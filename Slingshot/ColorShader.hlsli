//VERTEX SHADER
struct VS_INPUT
{
    float4 pos : POSITION0;
    float4 color : COLOR0;
};

struct VS_OUTPUT
{
    float4 pos : SV_Position;
    float4 color : COLOR0;
};

//PIXEL SHADER
struct PS_INPUT
{
    float4 pos : SV_Position;
    float4 color : COLOR0;
};

struct PS_OUTPUT
{
    float4 color : SV_Target0;
};
