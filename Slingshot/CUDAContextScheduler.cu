#include "CUDAContextScheduler.cuh"

namespace HC {
	__global__ void k_ComputeSurfacePixels(ComputeVertex* buffer, int nPixels, int surfaceW, int surfaceH) {
		
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid > nPixels) {
			return;
		}

		int pixelY = tid / surfaceW;
		int pixelX = tid - pixelY * surfaceW;
		float r = (float)pixelX / (float)surfaceW;;
		float g = (float)pixelY / (float)surfaceH;
		float b = 0.2f;
		float4 vColor { r,g,b,1.0 };
		buffer[tid].position = float4 { pixelX, pixelY, 0.5, 1.0 };
		buffer[tid].color = vColor;
	}

	//Compute GFX Test
	__global__ void k_ComputeBasicTriangle(ComputeVertex* buffer) {

		buffer[0].position = float4{ 0.0f, 0.0f, 1.0f, 1.0f };
		buffer[0].color = float4{ 1.0f, 0.0f, 0.0f, 1.0f };
		
		buffer[1].position = float4{ 1000.0f, 0.0f, 1.0f, 1.0f };
		buffer[1].color = float4{ 0.0f, 1.0f, 0.0f, 1.0f };

		buffer[2].position = float4{ 0.0f, 0.0005f, 1.0f, 1.0f };
		buffer[2].color = float4{ 0.0f, 0.0f, 1.0f, 1.0f };
	}

	//Pending compute concurrency implementation through CUDA streams (local async engines = 6)

	__host__ void InvokeCBTKernel(ComputeVertex** buffer) {
		k_ComputeBasicTriangle << <1, 1 >> > (*buffer);
	}

	__host__ void InvokeCSPKernel(ComputeVertex** surfaceBuffer, size_t* bufferSize, int surfaceW, int surfaceH) {

		int nPixels = surfaceW * surfaceH;

		int CTASize = 64;
		int gridSize = nPixels / CTASize + 1;

		//ComputeVertex* h_surfaceBuffer = (ComputeVertex*)malloc(*bufferSize);

		k_ComputeSurfacePixels << <gridSize, CTASize, 0, 0 >> >
			(*surfaceBuffer, nPixels, surfaceW, surfaceH);

		//ProfileCUDA(cudaMemcpy(h_surfaceBuffer, *surfaceBuffer, *bufferSize, cudaMemcpyDeviceToHost));
		//GenPPMFile("GfxExp", h_surfaceBuffer, surfaceW, surfaceH);
		//free(h_surfaceBuffer);
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

	__host__ void GenPPMFile(const char* fileName, HC::ComputeVertex* buffer, const int imgW, const int imgH) {
		std::string fn = std::string("./") + fileName + ".ppm";
		std::ofstream ofsGpu(fn.c_str(), std::ios::out | std::ios::binary);
		ofsGpu << "P6\n" << imgW << " " << imgH << "\n255\n";
		int nPixels = imgW * imgH;
		for (int i = 0; i < nPixels; ++i) {
			float4 v = buffer[i].color;
			int r = int(255.99f * v.x * v.w);
			int g = int(255.99f * v.y * v.w);
			int b = int(255.99f * v.z * v.w);
			ofsGpu << (unsigned char)r << (unsigned char)g << (unsigned char)b;
		}
		ofsGpu.close();
	}
}