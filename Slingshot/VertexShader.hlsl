float4 main( float4 pos : POSITION ) : SV_POSITION
{
	return pos;
}

//struct	VS_In
//{
//	uint vertexId : SV_VertexID;
//};
//
//struct VS_Out
//{
//	float4 pos : SV_Position;
//	float4 color : color;
//};
//
//VS_Out VS_main(float4 pos : POSITION) : SV_POSITION
//{
//	VS_Out output;
//	if (input.vertexId == 0) {
//		output.pos = float4(0.0, 0.5, 0.5, 1.0);
//	}
//	else if (input.vertexId == 2) {
//		output.pos = float4(0.5, -0.5, 0.5, 1.0);
//	}
//	else if (input.vertexId == 1) {
//		output.pos = float4(-0.5, -0.5, 0.5, 1.0);
//	}
//	output.color = clamp(output.pos, 0, 1);
//
//	return output;
//}