struct GS_COALESCENT
{
	float4 pos : SV_POSITION;
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
        GS_COALESCENT element;
		element.pos = input[i].pos;
        element.color = input[i].color;
        gsStream.Append(element);
    }
}