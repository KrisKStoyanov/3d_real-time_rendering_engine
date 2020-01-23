#pragma once
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cuda_d3d11_interop.h>
#include <d3dcompiler.h>

#include <dxgi1_6.h>
#include <DirectXMath.h>

#include "GUIConsole.h"

#define ProfileCUDA(val) CheckError((val),#val, __FILE__, __LINE__)

namespace HC {

	class vec3 {
	public:
		__host__ __device__ vec3() { e[0] = 0.0f, e[1] = 0.0f, e[2] = 0.0f; }
		__host__ __device__ vec3(const float v) { e[0] = v, e[1] = v, e[2] = v; }
		__host__ __device__ vec3(const float e0, float e1, float e2) { e[0] = e0, e[1] = e1, e[2] = e2; }
		__host__ __device__ vec3(const vec3& v) { e[0] = v.x(), e[1] = v.y(), e[2] = v.z(); }

		__host__ __device__ inline void x(const float v) { e[0] = v; }
		__host__ __device__ inline void y(const float v) { e[1] = v; }
		__host__ __device__ inline void z(const float v) { e[2] = v; }

		__host__ __device__ inline float x() const { return e[0]; }
		__host__ __device__ inline float y() const { return e[1]; }
		__host__ __device__ inline float z() const { return e[2]; }

		__host__ __device__ inline void r(const float v) { e[0] = v; }
		__host__ __device__ inline void g(const float v) { e[1] = v; }
		__host__ __device__ inline void b(const float v) { e[2] = v; }

		__host__ __device__ inline float r() const { return e[0]; }
		__host__ __device__ inline float g() const { return e[1]; }
		__host__ __device__ inline float b() const { return e[2]; }

	private:
		float e[3];
	};

	class vec4 {
	public:
		__host__ __device__ vec4() { e[0] = 0.0f, e[1] = 0.0f, e[2] = 0.0f, e[3] = 1.0f; }
		__host__ __device__ vec4(const float v) { e[0] = v, e[1] = v, e[2] = v, e[3] = 1.0f; }
		__host__ __device__ vec4(const float e0, float e1, float e2, float e3) { e[0] = e0, e[1] = e1, e[2] = e2, e[3] = e3; }
		__host__ __device__ vec4(const vec4& v) { e[0] = v.x(), e[1] = v.y(), e[2] = v.z(), e[3] = v.w(); }

		__host__ __device__ inline void x(const float v) { e[0] = v; }
		__host__ __device__ inline void y(const float v) { e[1] = v; }
		__host__ __device__ inline void z(const float v) { e[2] = v; }
		__host__ __device__ inline void w(const float v) { e[3] = v; }

		__host__ __device__ inline float x() const { return e[0]; }
		__host__ __device__ inline float y() const { return e[1]; }
		__host__ __device__ inline float z() const { return e[2]; }
		__host__ __device__ inline float w() const { return e[3]; }

		__host__ __device__ inline void r(const float v) { e[0] = v; }
		__host__ __device__ inline void g(const float v) { e[1] = v; }
		__host__ __device__ inline void b(const float v) { e[2] = v; }
		__host__ __device__ inline void a(const float v) { e[3] = v; }

		__host__ __device__ inline float r() const { return e[0]; }
		__host__ __device__ inline float g() const { return e[1]; }
		__host__ __device__ inline float b() const { return e[2]; }
		__host__ __device__ inline float a() const { return e[3]; }

	private:
		float e[4];
	};

	__host__ __device__ inline vec3 operator+ (const vec3 lhs, const vec3 rhs) { vec3 v(lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z()); return v; }
	__host__ __device__ inline vec3 operator* (const float lhs, const vec3 rhs) { vec3 v(lhs * rhs.x(), lhs * rhs.y(), lhs * rhs.z()); return v; }
	__host__ __device__ inline bool operator== (const vec3 lhs, const vec3 rhs) { return (lhs.x() == rhs.x() || lhs.y() == rhs.y() || lhs.z() == rhs.z()); }
	__host__ __device__ inline bool operator!= (const vec3 lhs, const vec3 rhs) { return !(lhs.x() == rhs.x() || lhs.y() == rhs.y() || lhs.z() == rhs.z()); }
	__host__ __device__ inline vec4 operator+ (const vec4 lhs, const vec4 rhs) { vec4 v(lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z(), lhs.w() + rhs.w()); return v; }
	__host__ __device__ inline vec4 operator* (const float lhs, const vec4 rhs) { vec4 v(lhs * rhs.x(), lhs * rhs.y(), lhs * rhs.z(), lhs * rhs.w()); return v; }

	__host__ __device__ inline float sqrt(const float v, double accuracy = 0.001)
	{
		float h = v / 2;

		float g = v / h;
		float s = (g + h) / 2;
		float d = h - s;
		float p;

		d < 0 ? d = -d : d;

		while (d > accuracy) {
			p = s;
			g = v / s;
			s = (g + s) / 2;
			d = p - s;
		}

		return s;
	}

	__host__ __device__ inline float mag(const vec3 v) { return (sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z())); }
	__host__ __device__ inline vec3 norm(const vec3 v) { float d = 1.0f / mag(v); return (vec3(v.x() * d, v.y() * d, v.z() * d)); }
	__host__ __device__ inline float dot(const vec3 a, const vec3 b) { return ((a.x() * b.x() + a.y() * b.y() + a.z() * b.z())); }
	__host__ __device__ inline vec3 cross(const vec3 a, const vec3 b) { return vec3((a.y() * b.z() - a.z() * b.y()),(a.z() * b.x() - a.x() * b.z()),(a.x() * b.y() - a.y() * b.x())); }

	class ray {
	public:
		__device__ ray() {}
		__device__ ray(const vec3& a, const vec3& b) { A = a, B = b; }
		__device__ inline vec3 origin() const { return A; }
		__device__ inline vec3 direction() const { return B; }
		__device__ inline vec3 intercept(const float t) const { return A + t * B; }
	private:
		vec3 A;
		vec3 B;
	};

	__host__ __device__ struct ComputeVertex {
		float4 position;
		float4 color;
	};

	__host__ void InvokeCSPKernel(ComputeVertex** surfaceBuffer, size_t* bufferSize, int surfaceW, int surfaceH);
	__host__ cudaDeviceProp QueryDeviceProperties(int dIndex);
	__host__ float ComputeSPEffectiveBandwith(int actThr, float kExecMs);
	__host__ float ComputeComputationalThroughput(int nFlops, int actThr, float kExecS);
	__host__ float ComputeHostToDeviceBandwith(unsigned int bytes, float elpsdMs);
	__host__ float ComputeDeviceToHostBandwith(unsigned int bytes, float elpsdMs);
	__host__ std::string GetPerformanceMetrics(
		float* kExecMs = NULL,
		float* efBw = NULL,
		float* compThr = NULL,
		float* htdBw = NULL,
		float* dthBw = NULL,
		unsigned int conSleepMs = 1000);
	__host__ __device__ void CheckError(cudaError_t result, char const* const func, const char* const file, int const line);
	__host__ void GenPPMFile(const char* fileName, HC::ComputeVertex* buffer, const int imgW, const int imgH);
	
	__device__ inline vec3 d_color(const ray& r) {
		vec3 rayDir = norm(r.direction());
		float t = 0.5f * (rayDir.y() + 1.0f);
		return (1.0f - t) * vec3(1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
	}

	__host__ void InvokeCBTKernel(ComputeVertex** buffer);

	//D3D11 Interop:
	class ComputeContext {
	public:
		__host__ inline ComputeContext(IDXGIAdapter* adapter, ID3D11Device* device, unsigned int numGpus = 1)
		{
			SetupGraphicsInteropContext(adapter, device, numGpus);
		}
		__host__ inline ~ComputeContext() {};

		__host__ inline bool SetupGraphicsInteropContext(IDXGIAdapter* adapter, ID3D11Device* device, unsigned int numGpus = 1)
		{
			numGpus > 1 ?
				m_DeviceIDs = GetD3D11DevicesIDs(device, numGpus) :
				m_DeviceIDs = GetD3D11DeviceID(adapter);

			return true;
		}

		__host__ void ProcessUnifiedTriangleBuffer(
			ID3D11Resource* d3d11Res,
			unsigned int flags)
		{
			cudaGraphicsResource* pInteropRes = NULL;
			ProfileCUDA(cudaGraphicsD3D11RegisterResource(&pInteropRes, d3d11Res, 4));
			ProfileCUDA(cudaGraphicsMapResources(1, &pInteropRes, 0));

			size_t* interopBufferSize = new size_t(0);
			ComputeVertex* interopBuffer = NULL;
			ProfileCUDA(cudaGraphicsResourceGetMappedPointer((void**)&interopBuffer, interopBufferSize, pInteropRes));
			InvokeCBTKernel((ComputeVertex**)&interopBuffer);

			ProfileCUDA(cudaGraphicsUnmapResources(1, &pInteropRes, 0));
			ProfileCUDA(cudaGraphicsUnregisterResource(pInteropRes));
		}

		__host__ void ProcessUnifiedSurfaceBuffer(
			ID3D11Resource* d3d11Res,
			unsigned int flags, 
			unsigned int surfaceW, 
			unsigned int surfaceH)
		{
			cudaGraphicsResource* pInteropRes = NULL;
			ProfileCUDA(cudaGraphicsD3D11RegisterResource(&pInteropRes, d3d11Res, 4));
			ProfileCUDA(cudaGraphicsMapResources(1, &pInteropRes, 0));

			size_t* interopBufferSize = new size_t(0);
			ComputeVertex* interopBuffer = NULL;
			ProfileCUDA(cudaGraphicsResourceGetMappedPointer((void**)&interopBuffer, interopBufferSize, pInteropRes));
			InvokeCSPKernel((ComputeVertex**)&interopBuffer, interopBufferSize, surfaceW, surfaceH);

			//CPU data conversion test
			//*iopBData = (ComputeVertex*)malloc(*surfaceBufferSize);
			//ProfileCUDA(cudaMemcpy(*iopBData, surfaceBuffer, *surfaceBufferSize, cudaMemcpyDeviceToHost));
			ProfileCUDA(cudaGraphicsUnmapResources(1, &pInteropRes, 0));
			ProfileCUDA(cudaGraphicsUnregisterResource(pInteropRes));
		}

		__host__ inline int* GetD3D11DeviceID(IDXGIAdapter* adapter) 
		{
			int* id = (int*)malloc(sizeof(int));
			ProfileCUDA(cudaD3D11GetDevice(id, adapter));
			return id;
		}

		//Setup for systems configured with SLI (predetermined GPU count)
		__host__ inline int* GetD3D11DevicesIDs(ID3D11Device* device, unsigned int numGpus = 1)
		{
			unsigned int numCudaDevices = 0;
			int* cudaDeviceIDs = new int[numGpus];

			ProfileCUDA(
				cudaD3D11GetDevices(
					&numCudaDevices,
					cudaDeviceIDs,
					sizeof(int) * numGpus, 
					device, 
					cudaD3D11DeviceList::cudaD3D11DeviceListAll));
			return cudaDeviceIDs;
		}

	private:
		int* m_DeviceIDs = NULL;
	};
}
