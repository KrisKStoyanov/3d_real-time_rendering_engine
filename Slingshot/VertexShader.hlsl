struct VS_INPUT {
	float4 pos : POSITION0;
    float4 color : COLOR0;
};

struct VS_OUTPUT {
	float4 pos : SV_POSITION;
	float4 color : COLOR0;
};

VS_OUTPUT main(VS_INPUT input)
{
	VS_OUTPUT output;
	output.pos = input.pos;
	output.color = input.color;
	return output;
}