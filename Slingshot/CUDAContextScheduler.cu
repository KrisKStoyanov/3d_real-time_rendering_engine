#include "CUDAContextScheduler.cuh"

namespace HC {
	__global__ void k_Render(vec3* frameBuffer, int nPixels, int areaW, int areaH) {
		
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid > nPixels) {
			return;
		}
		int pixelY = areaH - tid / areaW;
		int pixelX = tid - (tid / areaW) * areaW;
		float r = (float)pixelX / (float)areaW;;
		float g = (float)pixelY / (float)areaH;
		float b = 0.2f;
		vec3 v(r, g, b);
		frameBuffer[tid] = v;
	}

	//Pending compute concurrency implementation through CUDA streams (local async engines = 6)

	__host__ bool InvokeRenderKernel(vec3*& screenBuffer, int areaW, int areaH) {

		int nPixels = areaW * areaH;

		//dim3 CTAsize(8, 8);
		//dim3 gridSize(areaW / CTAsize.x + 1, areaH / CTAsize.y + 1);

		int CTASize = 64;
		int gridSize = nPixels / CTASize + 1;

		size_t fBufSize = nPixels * sizeof(vec3);

		screenBuffer = (vec3*)(malloc(fBufSize));
		vec3* d_FBuf;
		
		vec3 rayOrigin = vec3(areaW / 2, areaH / 2, 0.0f);
		
		int nDevices;
		ProfileCUDA(cudaGetDeviceCount(&nDevices));
		const int n = nDevices;
		cudaDeviceProp* dProps = new cudaDeviceProp[n];
		int asyncEngines = 0;
#if defined(_DEBUG)
		std::string dPropsStr = "Devices:\n-------";
#endif
		for (int i = 0; i < nDevices; ++i) {
			dProps[i] = QueryDeviceProperties(i);
#if defined(_DEBUG)
			dPropsStr.append(
				"\nDevice ID: " + i +
				std::string("\nDevice Name: ") + dProps[i].name +
				"\nMemory Clock Rate (KHz): " + std::to_string(dProps[i].memoryClockRate) +
				"\nMemory Bus Width (bits): " + std::to_string(dProps[i].memoryBusWidth) +
				"\nPeak Memory Bandwith (GB/s): " + std::to_string(2.0 * dProps[i].memoryClockRate * (dProps[i].memoryBusWidth / 8) / 1.0e6)
			);
#endif
			if (dProps[i].asyncEngineCount > asyncEngines) {
				asyncEngines = dProps[i].asyncEngineCount;
			}
		}
#if defined(_DEBUG)
		StreamOutputToConsole(dPropsStr.c_str());
#endif

		//Experimental: (override asyncEngines to 0 to return to default implementation)
		asyncEngines = 0;
		//----------------
		if (asyncEngines) {
			const int nEngines = asyncEngines;
			cudaStream_t* d_Streams = new cudaStream_t[nEngines];
			
			size_t d_FbufFeatureSize = fBufSize / nEngines;

			for (int i = 0; i < nEngines; ++i) {
				ProfileCUDA(cudaStreamCreate(&d_Streams[i]));
			}
			
			ProfileCUDA(cudaMallocHost((void**)&d_FBuf, fBufSize));

#if defined(_DEBUG)
			cudaEvent_t startK, stopK;
			ProfileCUDA(cudaEventCreate(&startK));
			ProfileCUDA(cudaEventCreate(&stopK));
			ProfileCUDA(cudaEventRecord(startK));
#endif

			for (int i = 0; i < nEngines; ++i) {
				int wOffset ;
				int hOffset;
				k_Render << <gridSize, CTASize, 0, d_Streams[i] >> > 
					(d_FBuf, nPixels, areaW, areaH);
			}
			
#if defined(_DEBUG)
			ProfileCUDA(cudaEventRecord(stopK));
			ProfileCUDA(cudaGetLastError());
			ProfileCUDA(cudaEventSynchronize(stopK));
			float kExecMs;
			ProfileCUDA(cudaEventElapsedTime(&kExecMs, startK, stopK));
			ProfileCUDA(cudaEventDestroy(startK));
			ProfileCUDA(cudaEventDestroy(stopK));
			float efBw = ComputeSPEffectiveBandwith(nPixels, kExecMs);
			float compThr = ComputeComputationalThroughput(18, nPixels, kExecMs / 1000);
#endif

#if defined(_DEBUG)
			cudaEvent_t startDMalloc, stopDMalloc;
			ProfileCUDA(cudaEventCreate(&startDMalloc));
			ProfileCUDA(cudaEventCreate(&stopDMalloc));
			ProfileCUDA(cudaEventRecord(startDMalloc));
#endif
			for (int i = 0; i < nEngines; ++i) {
				ProfileCUDA(cudaMemcpyAsync(screenBuffer, d_FBuf, fBufSize, cudaMemcpyDeviceToHost, d_Streams[i]));
			}
#if defined(_DEBUG)
			ProfileCUDA(cudaEventRecord(stopDMalloc));
			ProfileCUDA(cudaEventSynchronize(stopDMalloc));
			float mallocElapsedMs;
			ProfileCUDA(cudaEventElapsedTime(&mallocElapsedMs, startDMalloc, stopDMalloc));
			ProfileCUDA(cudaEventDestroy(startDMalloc));
			ProfileCUDA(cudaEventDestroy(stopDMalloc));
			float dthBw = ComputeDeviceToHostBandwith(sizeof(float), mallocElapsedMs);
			GetPerformanceMetrics(&kExecMs, &efBw, &compThr, NULL, &dthBw);
#endif
			for (int i = 0; i < nEngines; ++i) {
				ProfileCUDA(cudaStreamDestroy(d_Streams[i]));
			}
		}
		//----------------
		//Default:
		//----------------
		else {
			ProfileCUDA(cudaMalloc((void**)&d_FBuf, fBufSize));

#if defined(_DEBUG)
			cudaEvent_t startK, stopK;
			ProfileCUDA(cudaEventCreate(&startK));
			ProfileCUDA(cudaEventCreate(&stopK));
			ProfileCUDA(cudaEventRecord(startK));
#endif
			k_Render << <gridSize, CTASize, 0, 0 >> > (d_FBuf, nPixels, areaW, areaH);
#if defined(_DEBUG)
			ProfileCUDA(cudaEventRecord(stopK));
			ProfileCUDA(cudaGetLastError());
			ProfileCUDA(cudaEventSynchronize(stopK));
			float kExecMs;
			ProfileCUDA(cudaEventElapsedTime(&kExecMs, startK, stopK));
			ProfileCUDA(cudaEventDestroy(startK));
			ProfileCUDA(cudaEventDestroy(stopK));
			float efBw = ComputeSPEffectiveBandwith(nPixels, kExecMs);
			float compThr = ComputeComputationalThroughput(18, nPixels, kExecMs / 1000);
#endif

#if defined(_DEBUG)
			cudaEvent_t startDMalloc, stopDMalloc;
			ProfileCUDA(cudaEventCreate(&startDMalloc));
			ProfileCUDA(cudaEventCreate(&stopDMalloc));
			ProfileCUDA(cudaEventRecord(startDMalloc));
#endif
			ProfileCUDA(cudaMemcpyAsync(screenBuffer, d_FBuf, fBufSize, cudaMemcpyDeviceToHost));
#if defined(_DEBUG)
			ProfileCUDA(cudaEventRecord(stopDMalloc));
			ProfileCUDA(cudaEventSynchronize(stopDMalloc));
			float mallocElapsedMs;
			ProfileCUDA(cudaEventElapsedTime(&mallocElapsedMs, startDMalloc, stopDMalloc));
			ProfileCUDA(cudaEventDestroy(startDMalloc));
			ProfileCUDA(cudaEventDestroy(stopDMalloc));
			float dthBw = ComputeDeviceToHostBandwith(sizeof(float), mallocElapsedMs);
			GetPerformanceMetrics(&kExecMs, &efBw, &compThr, NULL, &dthBw);
#endif
		}
		//----------------

		GenPPMFile("GfxExp", screenBuffer, areaW, areaH);

		//free(screenBuffer);

		if (asyncEngines) {
			ProfileCUDA(cudaFreeHost(d_FBuf));
		}
		else {
			ProfileCUDA(cudaFree(d_FBuf));
		}

		return true;
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

	__host__ cudaDeviceProp QueryDeviceProperties(int dIndex) {
		cudaDeviceProp dProps;
		ProfileCUDA(cudaGetDeviceProperties(&dProps, dIndex));
		return dProps;
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

	__host__ std::string GetPerformanceMetrics(
		float* kExecMs, 
		float* efBw, 
		float* compThr, 
		float* htdBw, 
		float* dthBw,
		unsigned int conSleepMs)
	{
		std::string perfStr;
		
		if (kExecMs) {
			std::string kExecStr = "Kernel Execution Speed (MS): " + std::to_string(*kExecMs);
			perfStr.append(kExecStr);
		}

		if (efBw) {
			std::string efBwStr = "\nEffective Bandwith (GB/s): " + std::to_string(*efBw);
			perfStr.append(efBwStr);
		}

		if (compThr) {
			std::string compThrStr = "\nComputation Throughput (FLOPS/s): " + std::to_string(*compThr);
			perfStr.append(compThrStr);
		}

		if (htdBw) {
			std::string htdBwStr = "\nHost to Device bandwith (GB/s): " + std::to_string(*htdBw);
			perfStr.append(htdBwStr);
		}

		if (dthBw) {
			std::string dthBwStr = "\nDevice to Host bandwith (GB/s): " + std::to_string(*dthBw);
			perfStr.append(dthBwStr);
		}
		
		StreamOutputToConsole(perfStr.c_str(), conSleepMs);
		return perfStr;
	}

	__host__ void GenPPMFile(const char* fileName, vec3* buffer, const int imgW, const int imgH) {
		std::string fn = std::string("./") + fileName + ".ppm";
		std::ofstream ofsGpu(fn.c_str(), std::ios::out | std::ios::binary);
		ofsGpu << "P6\n" << imgW << " " << imgH << "\n255\n";
		int nPixels = imgW * imgH;
		for (int i = 0; i < nPixels; ++i) {
			vec3 v = buffer[i];
			int r = int(255.99f * v.r());
			int g = int(255.99f * v.g());
			int b = int(255.99f * v.b());
			ofsGpu << (unsigned char)r << (unsigned char)g << (unsigned char)b;
		}
		ofsGpu.close();
	}
}