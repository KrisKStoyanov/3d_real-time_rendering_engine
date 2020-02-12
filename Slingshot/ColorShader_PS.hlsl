#include "ColorShader.hlsli"

PS_OUTPUT main(PS_INPUT input)
{
    PS_OUTPUT output;
    output.color = input.color;
    return output;
}