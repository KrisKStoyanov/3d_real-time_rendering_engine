#include "CUDAContextScheduler.cuh"

namespace HC {
	__global__ void k_Render(vec3* frameBuffer, const int areaW, const int areaH) {
		int tidX = threadIdx.x + blockIdx.x * blockDim.x;
		int tidY = threadIdx.y + blockIdx.y * blockDim.y;
		if ((tidX >= areaW) || (tidY >= areaH)) {
			return;
		}
		int pId = tidY * areaW + tidX;
		vec3 v = vec3((float(tidX) / areaW), (float(tidY) / areaH), 0.2f);
		frameBuffer[pId] = v;
	}

	void ScheduleRenderKernel(int areaWidth, int areaHeight, dim3 CTAsize) {

		dim3 gridSize(1280 / CTAsize.x + 1, 720 / CTAsize.y + 1);

		int numPixels = areaWidth * areaHeight;
		size_t frameBufferSize = numPixels * sizeof(vec3);
		vec3* deviceFrameBuffer;

		checkCudaErrors(cudaMallocManaged((void**)&deviceFrameBuffer, frameBufferSize));
		k_Render << <gridSize, CTAsize >> > (deviceFrameBuffer, areaWidth, areaHeight);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		std::ofstream ofs("./cudaRaytraceGfx.ppm", std::ios::out | std::ios::binary);
		ofs << "P6\n" << areaWidth << " " << areaHeight << "\n255\n";
		for (int yOffset = areaHeight - 1; yOffset >= 0; --yOffset) {
			for (int xOffset = 0; xOffset < areaWidth; ++xOffset) {
				size_t pixelId = yOffset * areaWidth + xOffset;
				vec3 v = deviceFrameBuffer[pixelId];
				int r = int(255.99f * v.r());
				int g = int(255.99f * v.g());
				int b = int(255.99f * v.b());
				std::cout << r << " " << g << " " << b << "\n";
				ofs << (unsigned char)r << (unsigned char)g << (unsigned char)b;
			}
		}
		ofs.close();

		checkCudaErrors(cudaFree(deviceFrameBuffer));
	}

	void CheckError(cudaError_t result, char const* const func, const char* const file, int const line) {
		if (result) {
			std::cerr << "CUDA Error " << static_cast<unsigned int>(result)
				<< ":\nFile: " << file << "\nLine:" << line << std::endl;
			cudaDeviceReset();
			exit(99);
		}
	}
}