#include "ColorShader.hlsli"

PS_OUTPUT main(PS_INPUT ps_input)
{
    PS_OUTPUT ps_output;
    ps_output.color = ps_input.color;
    return ps_output;
}