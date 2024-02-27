__global__  void sinewave_vbo_kernel(float4 *pos,unsigned int width, unsigned int height,float animeTime)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = x/(float) width;

	float v = y/(float) height;

	u = u * 2.0 - 1.0;
	v = v * 2.0 - 1.0;


	float frequency = 4.0f;

	float w = sinf(frequency * u + animeTime) * cosf(frequency * v + animeTime) * 0.5f;

	pos[y *width+x] = make_float4(u,w,v,1.0);

}

void launchCudaKernel(float4 *pos,unsigned int mesh_width,unsigned int mesh_height,float time)
{
	dim3 block(8,8,1);

	dim3 grid(mesh_width / block.x, mesh_height/block.y,1.0);

	sinewave_vbo_kernel<<<grid,block>>>(pos,mesh_width,mesh_height,time);
}

