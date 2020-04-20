#include "DepthMap.hlsli"

cbuffer PerFrameBuffer : register(b0)
{
    float4x4 viewMatrix[6];
    float4x4 projectionMatrix;
}

[maxvertexcount(18)]
void main(triangle GS_INPUT gs_input[3] : SV_POSITION, inout TriangleStream<PS_INPUT> gs_output)
{   
    [unroll]
    for (int rtNum = 0; rtNum < 6; ++rtNum)
    {
        {
            PS_INPUT output = (PS_INPUT)0;
            output.RTAIndex = rtNum;
            
            [unroll]
            for (int vertNum = 0; vertNum < 3; ++vertNum)
            {
                output.position = mul(gs_input[vertNum].position, mul(viewMatrix[rtNum], projectionMatrix));
                gs_output.Append(output);
            }
            gs_output.RestartStrip();
        }
    }
}