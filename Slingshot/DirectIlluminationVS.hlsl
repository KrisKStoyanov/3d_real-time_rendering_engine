#include "DirectIllumination.hlsli"

cbuffer PerFrameBuffer : register(b0)
{
    float4x4 worldMatrix;
    float4x4 viewMatrix;
    float4x4 projMatrix;
};

float4 main( float4 pos : POSITION ) : SV_POSITION
{
	return pos;
}