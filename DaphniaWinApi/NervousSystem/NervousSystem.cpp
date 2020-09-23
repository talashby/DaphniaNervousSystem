#include "NervousSystem.h"
#include <array>
#include <vector>
#include <thread>
#include <atomic>
#include <assert.h>

// constants

constexpr int32_t ReinforcementForConditionedReflex = 10;

//

NervousSystem* s_nervousSystem = nullptr;
std::vector<std::thread> s_threads;
std::atomic<bool> s_isSimulationRunning = false;
std::atomic<uint64_t> s_time = 0; // absolute universe time
std::atomic<int32_t> s_waitThreadsCount = 0; // thread synchronization variable

//std::array s_eyeNetwork[]

void NervousSystemThread(int32_t threadNum);

void NervousSystem::Init()
{
	if (s_nervousSystem)
	{
		delete s_nervousSystem;
	}

	s_nervousSystem = new NervousSystem();
	s_threads.resize(1);
}

NervousSystem* NervousSystem::Instance()
{
	return s_nervousSystem;
}

void NervousSystem::StartSimulation(uint64_t timeOfTheUniverse)
{
	s_isSimulationRunning = true;
	s_time = timeOfTheUniverse;
	s_waitThreadsCount = s_threads.size();
	for (uint16_t ii = 0; ii < s_threads.size(); ++ii)
	{
		s_threads[ii] = std::thread(NervousSystemThread, ii);
	}
}

void NervousSystem::StopSimulation()
{
	if (!s_isSimulationRunning)
	{
		assert(false);
		return;
	}
	s_isSimulationRunning = false;
	for (uint16_t ii = 0; ii < s_threads.size(); ++ii)
	{
		s_threads[ii].join();
	}
}

bool NervousSystem::IsSimulationRunning() const
{
	return s_isSimulationRunning;
}

void NervousSystem::NextTick(uint64_t timeOfTheUniverse)
{
	if (!s_waitThreadsCount) // Is Tick Finished
	{
		if (timeOfTheUniverse > s_time)
		{
			if (0 < m_reinforcementLevel)
			{
				m_reinforcementLevelLast = m_reinforcementLevel;
				--m_reinforcementLevel;
			}
			s_waitThreadsCount = s_threads.size();
			++s_time;
		}
	}
}

void NervousSystem::PhotonReceived(uint8_t m_posX, uint8_t m_posY, PPh::EtherColor m_color)
{
}

void NervousSystemThread(int32_t threadNum)
{
	while (s_isSimulationRunning)
	{
		int32_t isTimeOdd = s_time % 2;
		// do something
		--s_waitThreadsCount;
		while (s_time % 2 == isTimeOdd)
		{
		}
	}
}