[numthreads(1, 1, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
}

//VertexShader vs = CompileShader(vs_5_0, VS_main());
//PixelShader ps = CompileShader(ps_5_0, PS_main());
//
//technique10 t0
//{
//	pass p0
//	{
//		SetVertexShader(vs);
//		SetGeometryShader(NULL);
//		SetPixelShader(ps);
//	};
//};