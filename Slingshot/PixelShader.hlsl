struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float4 col : COLOR;
};

struct PS_OUTPUT
{
	float4 col : SV_Target;
};

PS_OUTPUT main(PS_INPUT input)
{
	PS_OUTPUT output;
	output.col = input.col;
	return output;
}