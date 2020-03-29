#pragma once

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdexcept>

//MS D3D API - COM Interface derived pointer handling 
namespace DX
{
	class com_exception : public std::exception
	{
	public:
		com_exception(HRESULT hr) : result(hr) {}

		virtual const char* what() const override
		{
			static char s_str[64] = {};
			sprintf_s(s_str, "Failure with HRESULT of %08X",
				static_cast<unsigned int>(result));
			return s_str;
		}

	private:
		HRESULT result;
	};

	//D3D API exception utility
	inline void ThrowIfFailed(HRESULT hr) {
		if (FAILED(hr)) {
			throw com_exception(hr);
		}
	}

	//D3D API dedicated memory dealloc utility
	template<class T>
	void SafeRelease(T** ppT) {
		if (*ppT) {
			(*ppT)->Release();
			*ppT = NULL;
		}
	}
}
