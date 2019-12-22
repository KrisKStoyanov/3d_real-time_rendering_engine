

struct GS_COALESCENT
{
	float4 pos : SV_Position;
    float4 color : COLOR0;
};

[maxvertexcount(27)]
void main(
	triangle GS_COALESCENT input[3],
	inout TriangleStream<GS_COALESCENT> gsStream
)
{    
	for (uint i = 0; i < 3; i++)
	{
        //GS_COALESCENT element1;
        //element1.pos = input[i].pos;
        //element1.color = input[i].color;
        //gsStream.Append(element1);
        
        GS_COALESCENT element1;
        element1.pos = input[i].pos + float4(-0.2, 0.0, 0.0, 0.0);
        element1.color = input[i].color;
        gsStream.Append(element1);
        
        GS_COALESCENT element2;
        element2.pos = input[i].pos + float4(0.0, 0.2, 0.0, 0.0);
        element2.color = input[i].color;
        gsStream.Append(element2);
        
        GS_COALESCENT element3;
        element3.pos = input[i].pos + float4(0.2, 0.0, 0.0, 0.0);
        element3.color = input[i].color;
        gsStream.Append(element3);
        gsStream.RestartStrip();
    }
}