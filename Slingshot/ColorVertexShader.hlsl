struct VS_INPUT
{
    float4 position : POSITION;
    float4 color : COLOR;
};

struct PS_INPUT
{
    float4 position : SV_Position;
    float4 color : COLOR;
};

PS_INPUT main(VS_INPUT vs_input)
{
    PS_INPUT vs_output;
    vs_output.position = vs_input.position;
    vs_output.color = vs_input.color;
    return vs_output;
}