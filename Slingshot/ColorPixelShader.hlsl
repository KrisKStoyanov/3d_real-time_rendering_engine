struct PS_INPUT
{
    float4 position : SV_Position;
    float4 color : COLOR;
};

struct PS_OUTPUT
{
    float4 color : SV_Target;
};

PS_OUTPUT main(PS_INPUT ps_input)
{
    PS_OUTPUT ps_output;
    ps_output.color = ps_input.color;
    return ps_output;
}