#pragma once
#include <Windows.h>

class Timer
{
public:
	Timer();
	void OnFrameStart();

	double m_timestep;
	double m_smoothstep;
	double m_time;
	int m_frameCount;

	long long m_startupTimestamp;
	long long m_previousFrameTimestamp;
	double m_period;
};