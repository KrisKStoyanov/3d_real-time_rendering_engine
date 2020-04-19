#include "DepthMap.hlsli"

cbuffer PerFrameBuffer : register(b0)
{
    float4x4 viewMatrix[6];
    float4x4 projectionMatrix;
}

[maxvertexcount(18)]
void main(triangle GS_INPUT gs_input[3] : SV_POSITION, inout TriangleStream<PS_INPUT> gs_output)
{   
    for (int rtvNum = 0; rtvNum < 6; ++rtvNum)
    {
        for (int i = 0; i < 3; i++)
        {
            PS_INPUT element;
            element.position = mul(gs_input[i].position, mul(viewMatrix[rtvNum], projectionMatrix));
            //element.depthPos = element.position;
            gs_output.Append(element);
        }
        gs_output.RestartStrip();
    }
}