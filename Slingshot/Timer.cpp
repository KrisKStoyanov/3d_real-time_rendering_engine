#include "Timer.h"

Timer::Timer() :
	m_timestep(0.0),
	m_time(0.0),
	m_smoothstep(0.0),
	m_smoothstepF(0.0f),
	m_frameCount(0)
{
	long long frequency;
	QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);
	m_period = 1.0f / static_cast<double>(frequency);

	QueryPerformanceCounter((LARGE_INTEGER*)&m_startupTimestamp);
	m_previousFrameTimestamp = m_startupTimestamp;
}

void Timer::OnFrameStart()
{
	++m_frameCount;

	long long timestamp;
	QueryPerformanceCounter((LARGE_INTEGER*)&timestamp);

	m_time = static_cast<double>(timestamp - m_startupTimestamp) * m_period;

	m_timestep = static_cast<double>(timestamp - m_previousFrameTimestamp)* m_period;
	m_previousFrameTimestamp = timestamp;

	m_smoothstep = m_smoothstep * 0.9 + m_timestep * 0.1;
	m_smoothstepF = static_cast<float>(m_smoothstep);
}
