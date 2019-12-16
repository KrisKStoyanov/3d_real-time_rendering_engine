#include "CUDAContextScheduler.cuh"

namespace HC {
	__global__ void k_Render(vec3* frameBuffer, const int areaW, const int areaH,
		vec3 rayOrigin) {
		int tidX = threadIdx.x + blockIdx.x * blockDim.x;
		int tidY = threadIdx.y + blockIdx.y * blockDim.y;
		if ((tidX >= areaW) || (tidY >= areaH)) {
			return;
		}
		int pId = tidY * areaW + tidX;
		float u = (float(tidX) / float(areaW));
		float v = (float(tidY) / float(areaH));
		ray r(rayOrigin, vec3(u , v , 0.0f));
		frameBuffer[pId] = d_color(r);
	}

	void ScheduleRenderKernel(int areaW, int areaH, dim3 CTAsize) {

		dim3 gridSize(1280 / CTAsize.x + 1, 720 / CTAsize.y + 1);

		int numPixels = areaW * areaH;
		size_t frameBufferSize = numPixels * sizeof(vec3);
		vec3* deviceFrameBuffer;
		vec3 rayOrigin = vec3(areaW / 2, areaH / 2, 0.0f);
		
		checkCudaErrors(cudaMallocManaged((void**)&deviceFrameBuffer, frameBufferSize));
		k_Render << <gridSize, CTAsize >> > (deviceFrameBuffer, areaW, areaH, rayOrigin);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		std::ofstream ofs("./cudaRaytraceGfx.ppm", std::ios::out | std::ios::binary);
		ofs << "P6\n" << areaW << " " << areaH << "\n255\n";
		for (int yOffset = areaH - 1; yOffset >= 0; --yOffset) {
			for (int xOffset = 0; xOffset < areaW; ++xOffset) {
				size_t pixelId = yOffset * areaW + xOffset;
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
			unsigned int errId = static_cast<unsigned int>(result);
			std::string errStr =
				std::string("CUDA Error: ") + std::to_string(errId) +
				std::string(":\nFile: ") + file +
				std::string("\nLine: ") + std::to_string(line);
			StreamOutputToConsole(errStr.c_str(), stderr, 3000);
			cudaDeviceReset();
			exit(99);
		}
	}
}