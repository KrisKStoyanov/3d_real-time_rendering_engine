

struct GS_COALESCENT
{
	float4 pos : SV_Position;
    float4 color : COLOR0;
};

[maxvertexcount(3)]
void main(
	triangle GS_COALESCENT input[3],
	inout TriangleStream<GS_COALESCENT> gsStream
)
{    
	for (uint i = 0; i < 3; i++)
	{
        GS_COALESCENT element1;
        element1.pos = input[i].pos;
        element1.color = input[i].color;
        gsStream.Append(element1);
    }   
}