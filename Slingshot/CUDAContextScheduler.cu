#include "CUDAContextScheduler.cuh"

namespace HC {
	__global__ void k_Render(vec3* frameBuffer, const int areaW, const int areaH,
		vec3 rayOrigin) {
		//Thread ID offset by CTA ID with blockdim number of threads inside the grid 
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

	__host__ void ScheduleRenderKernel(int areaW, int areaH) {

		dim3 CTAsize(8, 8);
		dim3 gridSize(1280 / CTAsize.x + 1, 720 / CTAsize.y + 1);

		int nPixels = areaW * areaH;
		size_t fBufSize = nPixels * sizeof(vec3);

		vec3* h_FBuf = (vec3*)(malloc(fBufSize));
		vec3* d_FBuf;
		
		vec3 rayOrigin = vec3(areaW / 2, areaH / 2, 0.0f);

		ProfileCUDA(cudaMalloc((void**)&d_FBuf, fBufSize));
		//ProfileCUDA(cudaMallocHost((void**)&d_FBuf, fBufSize));

#if defined(_DEBUG)
		QueryDeviceProperties();
		cudaEvent_t startK, stopK;
		ProfileCUDA(cudaEventCreate(&startK));
		ProfileCUDA(cudaEventCreate(&stopK));
		ProfileCUDA(cudaEventRecord(startK));
#endif
		k_Render << <gridSize, CTAsize >> > (d_FBuf, areaW, areaH, rayOrigin);
#if defined(_DEBUG)
		ProfileCUDA(cudaEventRecord(stopK));
		ProfileCUDA(cudaGetLastError());
		ProfileCUDA(cudaEventSynchronize(stopK));
		float kExecMs;
		ProfileCUDA(cudaEventElapsedTime(&kExecMs, startK, stopK));
		ProfileCUDA(cudaEventDestroy(startK));
		ProfileCUDA(cudaEventDestroy(stopK));
		int actThreads = gridSize.x * CTAsize.x * areaW + gridSize.y * CTAsize.y * areaH;
		float efBw = ComputeSPEffectiveBandwith(actThreads, kExecMs);
		float compThr = ComputeComputationalThroughput(18, actThreads, kExecMs/1000);
#endif

#if defined(_DEBUG)
		cudaEvent_t startDMalloc, stopDMalloc;
		ProfileCUDA(cudaEventCreate(&startDMalloc));
		ProfileCUDA(cudaEventCreate(&stopDMalloc));
		ProfileCUDA(cudaEventRecord(startDMalloc));
#endif
		ProfileCUDA(cudaMemcpy(h_FBuf, d_FBuf, fBufSize, cudaMemcpyDeviceToHost));
#if defined(_DEBUG)
		ProfileCUDA(cudaEventRecord(stopDMalloc));
		ProfileCUDA(cudaEventSynchronize(stopDMalloc));
		float mallocElapsedMs;
		ProfileCUDA(cudaEventElapsedTime(&mallocElapsedMs, startDMalloc, stopDMalloc));
		ProfileCUDA(cudaEventDestroy(startDMalloc));
		ProfileCUDA(cudaEventDestroy(stopDMalloc));
		float dthBw = ComputeDeviceToHostBandwith(sizeof(float), mallocElapsedMs);
		GetPerformanceMetrics(kExecMs, efBw, compThr, 0.0f, dthBw);
#endif

		std::ofstream ofs("./cudaRaytraceGfx.ppm", std::ios::out | std::ios::binary);
		ofs << "P6\n" << areaW << " " << areaH << "\n255\n";
		for (int yOffset = areaH - 1; yOffset >= 0; --yOffset) {
			for (int xOffset = 0; xOffset < areaW; ++xOffset) {
				size_t pixelId = yOffset * areaW + xOffset;
				vec3 v = h_FBuf[pixelId];
				int r = int(255.99f * v.r());
				int g = int(255.99f * v.g());
				int b = int(255.99f * v.b());
				std::cout << r << " " << g << " " << b << "\n";
				ofs << (unsigned char)r << (unsigned char)g << (unsigned char)b;
			}
		}
		ofs.close();

		free(h_FBuf);
		ProfileCUDA(cudaFree(d_FBuf));
		//ProfileCUDA(cudaFreeHost(d_FBuf));
	}

	__host__ __device__ void CheckError(cudaError_t result, char const* const func, const char* const file, int const line) {
#if defined(_DEBUG)
		if (result) {
			unsigned int errId = static_cast<unsigned int>(result);
			const char* errName = cudaGetErrorName(result);
			const char* errDesc = cudaGetErrorString(result);
			std::string errStr =
				std::string("CUDA Error: ") + std::to_string(errId) + "\n" +
				std::string(errName) + ": " + std::string(errDesc) +
				std::string("\nFile: ") + file +
				std::string("\nLine: ") + std::to_string(line);
			
			cudaError_t resetErr = cudaDeviceReset();
			if (resetErr) {
				std::string resetErrStr =
					std::string("CUDA Reset Error: ") + std::to_string(errId) + "\n" +
					std::string(errName) + ": " + std::string(errDesc) +
					std::string("\nFile: ") + file +
					std::string("\nLine: ") + std::to_string(line);
				errStr.append(resetErrStr);
			}
			StreamOutputToConsole(errStr.c_str(), 3000, stderr);
			exit(99);
		}
#endif
	}

	__host__ std::string QueryDeviceProperties() {
		int nDevices;
		ProfileCUDA(cudaGetDeviceCount(&nDevices));
		const int n = nDevices;
		cudaDeviceProp* dProps = new cudaDeviceProp[n];
		std::string dPropsStr = "Devices:\n-------";
		for (int i = 0; i < nDevices; ++i) {
			ProfileCUDA(cudaGetDeviceProperties(&dProps[i], i));
			dPropsStr.append(
				"\nDevice ID: " + i +
				std::string("\nDevice Name: ") + dProps[i].name +
				"\nMemory Clock Rate (KHz): " + std::to_string(dProps[i].memoryClockRate) +
				"\nMemory Bus Width (bits): " + std::to_string(dProps[i].memoryBusWidth) +
				"\nPeak Memory Bandwith (GB/s): " + std::to_string(2.0 * dProps[i].memoryClockRate * (dProps[i].memoryBusWidth / 8) / 1.0e6)
			);
		}
		StreamOutputToConsole(dPropsStr.c_str());
		return dPropsStr;
	}

	__host__ float ComputeSPEffectiveBandwith(int actThr, float kExecMs)
	{
		return (actThr * sizeof(float) * 3 / kExecMs / 1e6);
	}

	__host__ float ComputeComputationalThroughput(int nFlops, int actThr, float kExecS)
	{
		return (nFlops * actThr / (kExecS * 1e9));
	}

	__host__ float ComputeHostToDeviceBandwith(unsigned int bytes, float elpsdMs)
	{
		return (bytes * 1e6 / elpsdMs);
	}

	__host__ float ComputeDeviceToHostBandwith(unsigned int bytes, float elpsdMs)
	{
		return (bytes * 1e6 / elpsdMs);
	}

	__host__ std::string GetPerformanceMetrics(float kExecMs, float efBw, float compThr, float htdBw, float dthBw)
	{
		std::string perfStr;
		
		if (kExecMs > 0.0f) {
			std::string kExecStr = "Kernel Execution Speed (MS): " + std::to_string(kExecMs);
			perfStr.append(kExecStr);
		}

		if (efBw > 0.0f) {
			std::string efBwStr = "\nEffective Bandwith (GB/s): " + std::to_string(efBw);
			perfStr.append(efBwStr);
		}

		if (compThr > 0.0f) {
			std::string compThrStr = "\nComputation Throughput (FLOPS/s): " + std::to_string(compThr);
			perfStr.append(compThrStr);
		}

		if (htdBw > 0.0f) {
			std::string htdBwStr = "\nHost to Device bandwith (GB/s): " + std::to_string(htdBw);
			perfStr.append(htdBwStr);
		}

		if (dthBw > 0.0f) {
			std::string dthBwStr = "\nDevice to Host bandwith (GB/s): " + std::to_string(dthBw);
			perfStr.append(dthBwStr);
		}
		
		StreamOutputToConsole(perfStr.c_str(), 3000);
		return perfStr;
	}
}