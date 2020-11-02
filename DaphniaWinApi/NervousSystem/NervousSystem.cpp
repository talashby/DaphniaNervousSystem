
#include "NervousSystem.h"

#include "Neurons.h"
#include <array>
#include <vector>
#include <thread>
#include <atomic>
#include <assert.h>
#include "ParallelPhysics/ServerProtocol.h"
#include "ParallelPhysics/ObserverClient.h"
#include <algorithm>
#include <utility>
#include <mutex>
#include <algorithm>
#include <xutility>

#define HIGH_PRECISION_STATS 1

// constants
constexpr uint32_t CONDITIONED_REFLEX_CONTAINER_NUM = 100; // units
constexpr int32_t REINFORCEMENT_FOR_CONDITIONED_REFLEX = 20000; // units
//

static NervousSystem* s_nervousSystem = nullptr;
static std::array<std::thread, 4> s_threads;
static std::array<std::pair<uint32_t, uint32_t>, s_threads.size()> s_threadNeurons; // [first neuron; last neuron)
static std::atomic<bool> s_isSimulationRunning = false;
static std::atomic<uint64_t> s_time = 0; // absolute universe time
static std::atomic<uint32_t> s_waitThreadsCount = 0; // thread synchronization variable
static std::atomic<uint32_t> s_reinforcementStorageCurrentTick = 0; // store reinforcements in current tick

// stats
static std::mutex s_statisticsMutex;
static std::vector<uint32_t> s_timingsNervousSystemThreads;
static std::vector<uint32_t> s_tickTimeMusNervousSystemThreads;
std::atomic<uint32_t> s_status; // NervousSystemStatus



constexpr static int EYE_COLOR_NEURONS_NUM =  PPh::GetObserverEyeSize()*PPh::GetObserverEyeSize();
static std::array<std::array<SensoryNeuronRed, PPh::GetObserverEyeSize()>, PPh::GetObserverEyeSize()> s_eyeNetworkRed;
static std::array<std::array<SensoryNeuronGreen, PPh::GetObserverEyeSize()>, PPh::GetObserverEyeSize()> s_eyeNetworkGreen;
static std::array<std::array<SensoryNeuronBlue, PPh::GetObserverEyeSize()>, PPh::GetObserverEyeSize()> s_eyeNetworkBlue;
static std::array<MotorNeuron, 3> s_motorNetwork; // 0 - forward, 1 - left, 2 - right
static std::array<ExcitationAccumulatorNeuron, EYE_COLOR_NEURONS_NUM*3 + s_motorNetwork.size()> s_excitationAccumulatorNetwork;
static std::array<ConditionedReflexNeuron, CONDITIONED_REFLEX_CONTAINER_NUM*CONDITIONED_REFLEX_PER_CONTAINER> s_conditionedReflexNetwork;
static std::array<PrognosticNeuron, CONDITIONED_REFLEX_CONTAINER_NUM*CONDITIONED_REFLEX_PER_CONTAINER> s_prognosticNetwork;
static std::array<ConditionedReflexContainerNeuron, CONDITIONED_REFLEX_CONTAINER_NUM> s_conditionedReflexContainerNetwork;
static ConditionedReflexCreatorNeuron s_conditionedReflexCreatorNeuron;

struct NetworksMetadata
{
	uint32_t m_beginNeuronNum;
	uint32_t m_endNeuronNum; // last+1
	uint64_t m_begin;
	uint64_t m_end;
	uint32_t m_size;
};
static std::array s_networksMetadata{
	NetworksMetadata{0, 0, (uint64_t)(&s_eyeNetworkRed[0][0]), (uint64_t)(&s_eyeNetworkRed[0][0]+EYE_COLOR_NEURONS_NUM), sizeof(SensoryNeuronRed)},
	NetworksMetadata{0, 0, (uint64_t)(&s_eyeNetworkGreen[0][0]), (uint64_t)(&s_eyeNetworkGreen[0][0] + EYE_COLOR_NEURONS_NUM), sizeof(SensoryNeuronGreen)},
	NetworksMetadata{0, 0, (uint64_t)(&s_eyeNetworkBlue[0][0]), (uint64_t)(&s_eyeNetworkBlue[0][0] + EYE_COLOR_NEURONS_NUM), sizeof(SensoryNeuronBlue)},
	NetworksMetadata{0, 0, (uint64_t)&s_motorNetwork[0], (uint64_t)(&s_motorNetwork[0]+s_motorNetwork.size()), sizeof(MotorNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_excitationAccumulatorNetwork[0], (uint64_t)(&s_excitationAccumulatorNetwork[0] + s_excitationAccumulatorNetwork.size()), sizeof(ExcitationAccumulatorNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_conditionedReflexContainerNetwork[0], (uint64_t)(&s_conditionedReflexContainerNetwork[0] + s_conditionedReflexContainerNetwork.size()), sizeof(ConditionedReflexContainerNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_conditionedReflexCreatorNeuron, (uint64_t)(&s_conditionedReflexCreatorNeuron + 1), sizeof(ConditionedReflexCreatorNeuron)}
};

namespace NSNamespace
{
	uint64_t GetNSTime()
	{
		return s_time;
	}

	uint32_t GetNeuronIndex(Neuron *neuron)
	{
		uint64_t neuronAddr = reinterpret_cast<uint64_t>(neuron);
		for (auto &metadata : s_networksMetadata)
		{
			if (neuronAddr >= metadata.m_begin && neuronAddr < metadata.m_end)
			{
				return (uint32_t)((neuronAddr - metadata.m_begin) / metadata.m_size);
			}
		}
		assert(false);
		return 0;
	}

	Neuron* GetNeuronInterface(uint32_t neuronId)
	{
		Neuron* neuron = nullptr;
		uint8_t neuronType = (neuronId >> 24) & 0xff;
		uint16_t neuronIndex = neuronId & 0xffffff;
		switch (neuronType)
		{
		case SensoryNeuron::GetTypeStatic():
		{
			assert(false);
		}
		break;
		case SensoryNeuronRed::GetTypeStatic():
		{
			assert(s_eyeNetworkRed.size());
			assert(neuronIndex < s_eyeNetworkRed.size()*s_eyeNetworkRed[0].size());
			uint16_t xx = neuronIndex / PPh::GetObserverEyeSize();
			uint16_t yy = neuronIndex - xx * PPh::GetObserverEyeSize();
			neuron = &s_eyeNetworkRed[xx][yy];
		}
		break;
		case SensoryNeuronGreen::GetTypeStatic():
		{
			assert(s_eyeNetworkGreen.size());
			assert(neuronIndex < s_eyeNetworkGreen.size()*s_eyeNetworkGreen[0].size());
			uint16_t xx = neuronIndex / PPh::GetObserverEyeSize();
			uint16_t yy = neuronIndex - xx * PPh::GetObserverEyeSize();
			neuron = &s_eyeNetworkGreen[xx][yy];
		}
		break;
		case SensoryNeuronBlue::GetTypeStatic():
		{
			assert(s_eyeNetworkBlue.size());
			assert(neuronIndex < s_eyeNetworkBlue.size()*s_eyeNetworkBlue[0].size());
			uint16_t xx = neuronIndex / PPh::GetObserverEyeSize();
			uint16_t yy = neuronIndex - xx * PPh::GetObserverEyeSize();
			neuron = &s_eyeNetworkBlue[xx][yy];
		}
		break;
		case MotorNeuron::GetTypeStatic():
			assert(neuronIndex < s_motorNetwork.size());
			neuron = &s_motorNetwork[neuronIndex];
			break;
		case ExcitationAccumulatorNeuron::GetTypeStatic():
			assert(neuronIndex < s_excitationAccumulatorNetwork.size());
			neuron = &s_excitationAccumulatorNetwork[neuronIndex];
			break;
		case ConditionedReflexNeuron::GetTypeStatic():
			assert(neuronIndex < s_conditionedReflexNetwork.size());
			neuron = &s_conditionedReflexNetwork[neuronIndex];
			break;
		case ConditionedReflexCreatorNeuron::GetTypeStatic():
			assert(neuronIndex == 0);
			neuron = &s_conditionedReflexCreatorNeuron;
			break;
		}
		assert(((uint64_t*)neuron)[0] > 0); // virtual table should exist
		return neuron;
	}

	ConditionedReflexCreatorNeuron* GetConditionedReflexCreatorNeuron()
	{
		return &s_conditionedReflexCreatorNeuron;
	}

	uint32_t GetAccumulatorExitationNeuronNum()
	{
		return s_excitationAccumulatorNetwork.size();
	}

	void AddReinforcement(uint32_t val)
	{
		s_reinforcementStorageCurrentTick.fetch_add(val, std::memory_order_relaxed);
	}

}

void NervousSystemThread(uint32_t threadNum);

void NervousSystem::Init()
{
	if (s_nervousSystem)
	{
		delete s_nervousSystem;
	}

	uint32_t firstNeuronNum = 0;
	for (auto &el : s_networksMetadata)
	{
		el.m_beginNeuronNum = firstNeuronNum;
		el.m_endNeuronNum = firstNeuronNum + (uint32_t)((el.m_end - el.m_begin) / el.m_size);
		firstNeuronNum = el.m_endNeuronNum;
	}

#ifdef HIGH_PRECISION_STATS
	s_timingsNervousSystemThreads.resize(s_threads.size());
	s_tickTimeMusNervousSystemThreads.resize(s_threads.size());
#endif

	static_assert(s_threads.size() > 1);
	uint32_t threadsNumSpecial = s_threads.size()-1; // last thread for ConditionedReflexCreatorNeuron
	uint32_t neuronsNum = s_networksMetadata[s_networksMetadata.size()-2].m_endNeuronNum;
	assert(s_networksMetadata.back().m_endNeuronNum - neuronsNum == 1); // last neuron should be ConditionedReflexCreatorNeuron
	uint32_t step = neuronsNum / threadsNumSpecial;
	uint32_t remain = neuronsNum - step * threadsNumSpecial;
	uint32_t posBegin = 0;
	for (uint32_t ii = 0; ii < s_threadNeurons.size()-1; ++ii)
	{
		std::pair<uint32_t, uint32_t> &pair = s_threadNeurons[ii];
		int32_t posEnd = posBegin + step;
		if (0 < remain)
		{
			++posEnd;
			--remain;
		}
		pair.first = posBegin;
		pair.second = posEnd;
		posBegin = posEnd;
	}
	s_threadNeurons.back().first = s_networksMetadata.back().m_beginNeuronNum; // last thread for ConditionedReflexCreatorNeuron
	s_threadNeurons.back().second = s_networksMetadata.back().m_endNeuronNum; // last thread for ConditionedReflexCreatorNeuron
	 
	uint32_t excitationAccumulatorNetworkIndex = 0;
	for (uint32_t ii=0; ii<EYE_COLOR_NEURONS_NUM; ++ii)
	{
		uint32_t id = ii;
		uint32_t type = SensoryNeuronRed::GetTypeStatic();
		type = type << 24;
		id |= type;
		s_excitationAccumulatorNetwork[excitationAccumulatorNetworkIndex++].Init(id);
	}
	for (uint32_t ii = 0; ii < EYE_COLOR_NEURONS_NUM; ++ii)
	{
		uint32_t id = ii;
		uint32_t type = SensoryNeuronGreen::GetTypeStatic();
		type = type << 24;
		id |= type;
		s_excitationAccumulatorNetwork[excitationAccumulatorNetworkIndex++].Init(id);
	}
	for (uint32_t ii = 0; ii < EYE_COLOR_NEURONS_NUM; ++ii)
	{
		uint32_t id = ii;
		uint32_t type = SensoryNeuronBlue::GetTypeStatic();
		type = type << 24;
		id |= type;
		s_excitationAccumulatorNetwork[excitationAccumulatorNetworkIndex++].Init(id);
	}
	for (uint32_t ii = 0; ii < s_motorNetwork.size(); ++ii)
	{
		uint32_t id = ii;
		uint32_t type = MotorNeuron::GetTypeStatic();
		type = type << 24;
		id |= type;
		s_excitationAccumulatorNetwork[excitationAccumulatorNetworkIndex++].Init(id);
	}
	for (uint32_t ii = 0; ii < s_conditionedReflexContainerNetwork.size(); ++ii)
	{
		s_conditionedReflexContainerNetwork[ii].Init(&s_conditionedReflexNetwork[ii],
			&s_conditionedReflexNetwork[ii*CONDITIONED_REFLEX_PER_CONTAINER] + CONDITIONED_REFLEX_PER_CONTAINER);
		assert(ii*CONDITIONED_REFLEX_PER_CONTAINER + CONDITIONED_REFLEX_PER_CONTAINER <= s_conditionedReflexNetwork.size());
	}
	assert(s_excitationAccumulatorNetwork.size() == excitationAccumulatorNetworkIndex);
	s_conditionedReflexCreatorNeuron.Init(&s_excitationAccumulatorNetwork[0], &s_excitationAccumulatorNetwork[0] + s_excitationAccumulatorNetwork.size(),
		&s_conditionedReflexNetwork[0], &s_conditionedReflexNetwork[0] + s_conditionedReflexNetwork.size(),
		&s_prognosticNetwork[0], &s_prognosticNetwork[0] + s_prognosticNetwork.size());

	s_nervousSystem = new NervousSystem();
}

NervousSystem* NervousSystem::Instance()
{
	return s_nervousSystem;
}

void NervousSystem::StartSimulation(uint64_t timeOfTheUniverse)
{
	s_isSimulationRunning = true;
	s_time = timeOfTheUniverse;
	m_lastTime = PPh::GetTimeMs();
	m_lastTimeUniverse = timeOfTheUniverse;

	for (const auto &metadata : s_networksMetadata)
	{
		for (uint32_t jj = metadata.m_beginNeuronNum; jj < metadata.m_endNeuronNum; ++jj)
		{
			Neuron *neuron = reinterpret_cast<Neuron*>(metadata.m_begin + (jj - metadata.m_beginNeuronNum) * metadata.m_size);
			neuron->Init();
		}
	}
	s_waitThreadsCount = (uint32_t)s_threads.size();
	for (uint16_t ii = 0; ii < s_threads.size(); ++ii)
	{
		s_threads[ii] = std::thread(NervousSystemThread, ii);
	}
}

void NervousSystem::StopSimulation()
{
	if (!s_isSimulationRunning)
	{
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

void NervousSystem::GetStatisticsParams(uint32_t &reinforcementLevelStat, uint32_t &reinforcementsCountStat, uint32_t &condReflCountStat,
	uint32_t &movingSpontaneousCount, uint32_t &condReflLaunched, int32_t &minConditionedTmp,
	uint32_t &minNervousSystemTiming, uint32_t &maxNervousSystemTiming,	uint32_t &conditionedReflexCreatorTiming) const
{
	std::lock_guard<std::mutex> guard(s_statisticsMutex);
	reinforcementLevelStat = m_reinforcementLevelStat;
	reinforcementsCountStat = m_reinforcementsCountStat;
	condReflCountStat = NSNamespace::GetConditionedReflexCreatorNeuron()->GetCondReflCountStat();
	movingSpontaneousCount = MotorNeuron::GetMovingSpontaneousCount();
	condReflLaunched = NSNamespace::GetConditionedReflexCreatorNeuron()->GetCondReflLaunchedStat();
	minConditionedTmp = m_minConditionedTmpStat;
	minNervousSystemTiming = std::numeric_limits<uint32_t>::max();
	maxNervousSystemTiming = 0;
	for (uint32_t ii = 0; ii < s_tickTimeMusNervousSystemThreads.size()-1; ++ii)
	{
		minNervousSystemTiming = std::min(s_tickTimeMusNervousSystemThreads[ii], minNervousSystemTiming);
		maxNervousSystemTiming = std::max(s_tickTimeMusNervousSystemThreads[ii], maxNervousSystemTiming);
	}
	conditionedReflexCreatorTiming = s_tickTimeMusNervousSystemThreads[s_tickTimeMusNervousSystemThreads.size() - 1];
}

int32_t NervousSystem::GetReinforcementCount() const
{
	return m_reinforcementsCountStat;
}

int32_t NervousSystem::GetReinforcementLevel() const
{
	return m_reinforcementLevelStat;
}

uint64_t NervousSystem::GetTime() const
{
	return s_time;
}

void NervousSystem::NextTick(uint64_t timeOfTheUniverse)
{
	while(s_waitThreadsCount) // Is Tick Finished
	{ }
	{
		if (timeOfTheUniverse > s_time)
		{
			m_reinforcementLevelLast = m_reinforcementLevel;
			m_reinforcementLevel += s_reinforcementStorageCurrentTick;
			s_reinforcementStorageCurrentTick = 0;
			if (0 < m_reinforcementLevel)
			{
				m_reinforcementLevel -= 1 + m_reinforcementLevelSub / (100*MILLISECOND_IN_QUANTS);
				++m_reinforcementLevelSub;
				std::lock_guard<std::mutex> guard(s_statisticsMutex);
				if (m_reinforcementZeroTouched && m_reinforcementLevel >= REINFORCEMENT_FOR_CONDITIONED_REFLEX && m_reinforcementLevelLast < REINFORCEMENT_FOR_CONDITIONED_REFLEX)
				{
					++m_reinforcementsCountStat;
					m_reinforcementZeroTouched = false;
				}
			}
			else
			{
				m_reinforcementLevel = 0;
				m_reinforcementZeroTouched = true;
				m_reinforcementLevelSub = 0;
			}
			m_reinforcementLevelStat = m_reinforcementLevel;
			s_waitThreadsCount = (uint32_t)s_threads.size();

			if (PPh::GetTimeMs() - m_lastTime >= 1000 && s_time != m_lastTimeUniverse)
			{
				m_quantumOfTimePerSecond = (uint32_t)(s_time - m_lastTimeUniverse);
#ifdef HIGH_PRECISION_STATS
				std::lock_guard<std::mutex> guard(s_statisticsMutex);
				for (uint32_t ii = 0; ii < s_timingsNervousSystemThreads.size(); ++ii)
				{
					if (s_timingsNervousSystemThreads[ii] > 0)
					{
						s_tickTimeMusNervousSystemThreads[ii] = s_timingsNervousSystemThreads[ii] / m_quantumOfTimePerSecond;
						s_timingsNervousSystemThreads[ii] = 0;
					}
				}
#endif
				m_lastTime = PPh::GetTimeMs();
				m_lastTimeUniverse = s_time;
			}

			++s_time;
		}
	}
}

void NervousSystem::PhotonReceived(uint8_t m_posX, uint8_t m_posY, PPh::EtherColor m_color)
{
	if (m_color.m_colorR > 0)
	{
		s_eyeNetworkRed[m_posX][m_posY].ExcitatorySynapse();
	}
	else if (m_color.m_colorG > 0)
	{
		s_eyeNetworkGreen[m_posX][m_posY].ExcitatorySynapse();
	}
	else if (m_color.m_colorB > 0)
	{
		s_eyeNetworkBlue[m_posX][m_posY].ExcitatorySynapse();
	}
}

bool NervousSystem::IsReinforcementLevelLow() const
{
	return m_reinforcementLevel <= 1000;
}

bool NervousSystem::IsReinforcementHappened() const
{
	return !m_reinforcementZeroTouched;
}

void NervousSystem::SetConditionedTmpStat(uint32_t val)
{
	std::lock_guard<std::mutex> guard(s_statisticsMutex);
	if (-1 == m_minConditionedTmpStat || (int32_t)val < m_minConditionedTmpStat)
	{
		m_minConditionedTmpStat = val;
	}
}

const char* NervousSystem::GetStatus() const
{
	NervousSystemStatus status = static_cast<NervousSystemStatus>(s_status.load());
	switch (status)
	{
	case NervousSystemStatus::Relaxing:
		return "Relaxing";
	case NervousSystemStatus::SpontaneousActivity:
		return "SpontaneousActivity";
	case NervousSystemStatus::ConditionedReflexProceed:
		return "ConditionedReflexProceed";
	default:
		assert(false);
		break;
	}
	
	return "Relaxing";
}

void NervousSystem::SetStatus(NervousSystemStatus status)
{
	s_status = static_cast<uint32_t>(status);
}

void NervousSystemThread(uint32_t threadNum)
{
	assert(s_threadNeurons.size() > threadNum);
	uint32_t beginNetworkMetadata = 0;
	for (const auto &metadata : s_networksMetadata)
	{
		if (s_threadNeurons[threadNum].first >= metadata.m_beginNeuronNum && s_threadNeurons[threadNum].first < metadata.m_endNeuronNum)
		{
			break;
		}
		++beginNetworkMetadata;
	}
	uint32_t endNetworkMetadata = 0;
	for (const auto &metadata : s_networksMetadata)
	{
		if (s_threadNeurons[threadNum].second >= metadata.m_beginNeuronNum && s_threadNeurons[threadNum].second < metadata.m_endNeuronNum)
		{
			++endNetworkMetadata;
			break;
		}
		++endNetworkMetadata;
	}
	assert(beginNetworkMetadata <= endNetworkMetadata);
	uint32_t endLocalNeuronNum = s_threadNeurons[threadNum].second;
	while (s_isSimulationRunning)
	{
#ifdef HIGH_PRECISION_STATS
		auto beginTime = std::chrono::high_resolution_clock::now();
#endif
		int32_t isTimeOdd = s_time % 2;
		uint32_t beginLocalNeuronNum = s_threadNeurons[threadNum].first;
		for (uint32_t ii = beginNetworkMetadata; ii < endNetworkMetadata; ++ii)
		{
			if (ii != beginNetworkMetadata)
			{
				beginLocalNeuronNum = s_networksMetadata[ii].m_beginNeuronNum;
			}
			for (uint32_t jj = beginLocalNeuronNum; jj < endLocalNeuronNum && jj < s_networksMetadata[ii].m_endNeuronNum; ++jj)
			{
				Neuron *neuron = reinterpret_cast<Neuron*>(s_networksMetadata[ii].m_begin + (jj - s_networksMetadata[ii].m_beginNeuronNum) * s_networksMetadata[ii].m_size);
				neuron->Tick();
			}
		}
#ifdef HIGH_PRECISION_STATS
		auto endTime = std::chrono::high_resolution_clock::now();
		auto dif = endTime - beginTime;
		s_timingsNervousSystemThreads[threadNum] += (uint32_t)std::chrono::duration_cast<std::chrono::microseconds>(dif).count();
#endif
		--s_waitThreadsCount;
		while (s_time % 2 == isTimeOdd && s_isSimulationRunning)
		{
		}
	}
}
