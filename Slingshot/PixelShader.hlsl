struct PS_INPUT
{
	float4 pos : SV_Position;
	float4 color : COLOR0;
};

struct PS_OUTPUT
{
	float4 color : SV_Target0;
};

PS_OUTPUT main(PS_INPUT input)
{
	PS_OUTPUT output;
	output.color = input.color;
	return output;
}