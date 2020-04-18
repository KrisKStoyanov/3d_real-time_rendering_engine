#include "DepthMap.hlsli"

cbuffer PerFrameBuffer : register(b0)
{
    float4x4 viewMatrix0;
    float4x4 viewMatrix1;
    float4x4 viewMatrix2;
    float4x4 viewMatrix3;
    float4x4 viewMatrix4;
    float4x4 viewMatrix5;
	
    float4x4 projectionMatrix;
}

[maxvertexcount(18)]
void main(triangle GS_COALESCENT gs_input[3] : SV_POSITION, inout TriangleStream<GS_COALESCENT> gs_output)
{
    float4x4 vpMatrix0 = mul(viewMatrix0, projectionMatrix);
    float4x4 vpMatrix1 = mul(viewMatrix1, projectionMatrix);
    float4x4 vpMatrix2 = mul(viewMatrix2, projectionMatrix);
    float4x4 vpMatrix3 = mul(viewMatrix3, projectionMatrix);
    float4x4 vpMatrix4 = mul(viewMatrix4, projectionMatrix);
    float4x4 vpMatrix5 = mul(viewMatrix5, projectionMatrix);
    
    for (int rtvNum = 0; rtvNum < 6; ++rtvNum)
    {
        for (int i = 0; i < 3; i++)
        {
            GS_COALESCENT element;
            element.position = mul(gs_input[i].position, vpMatrix0); //pick appropriate matrix
            element.depthPos = element.position;
            gs_output.Append(element);
        }
        gs_output.RestartStrip();
    }
}