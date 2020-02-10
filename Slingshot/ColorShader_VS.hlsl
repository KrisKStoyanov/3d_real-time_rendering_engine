#include "ColorShader.hlsli"

PS_INPUT main(VS_INPUT input)
{
    PS_INPUT output;
    output.position = input.position;
    output.color = input.color;
    return output;
}