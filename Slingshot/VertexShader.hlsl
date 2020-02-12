struct VS_INPUT {
	float4 pos : POSITION0;
    float4 color : COLOR0;
};

struct PS_INPUT
{
    float4 pos : SV_Position;
    float4 color : COLOR0;
};

PS_INPUT main(VS_INPUT vs_input)
{
    PS_INPUT vs_output;
    vs_output.pos = vs_input.pos;
    vs_output.color = vs_input.color;
    return vs_output;
}