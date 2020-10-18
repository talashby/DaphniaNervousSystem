
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

// constants
constexpr uint32_t CONDITIONED_REFLEX_CONTAINER_NUM = 100; // units
constexpr int32_t REINFORCEMENT_FOR_CONDITIONED_REFLEX = 20000; // units
//

static NervousSystem* s_nervousSystem = nullptr;
static std::array<std::thread, 3> s_threads;
static std::array<std::pair<uint32_t, uint32_t>, s_threads.size()> s_threadNeurons; // [first neuron; last neuron)
static std::atomic<bool> s_isSimulationRunning = false;
static std::atomic<uint64_t> s_time = 0; // absolute universe time
static std::atomic<uint32_t> s_waitThreadsCount = 0; // thread synchronization variable
static std::atomic<uint32_t> s_reinforcementStorageCurrentTick = 0; // store reinforcements in current tick

static std::mutex s_statisticsMutex;







constexpr static int EYE_COLOR_NEURONS_NUM =  PPh::GetObserverEyeSize()*PPh::GetObserverEyeSize();
static std::array<std::array<SensoryNeuronRed, PPh::GetObserverEyeSize()>, PPh::GetObserverEyeSize()> s_eyeNetworkRed;
static std::array<std::array<SensoryNeuronGreen, PPh::GetObserverEyeSize()>, PPh::GetObserverEyeSize()> s_eyeNetworkGreen;
static std::array<std::array<SensoryNeuronBlue, PPh::GetObserverEyeSize()>, PPh::GetObserverEyeSize()> s_eyeNetworkBlue;
static std::array<MotorNeuron, 3> s_motorNetwork; // 0 - forward, 1 - left, 2 - right
static std::array<ExcitationAccumulatorNeuron, EYE_COLOR_NEURONS_NUM*3 + s_motorNetwork.size()> s_excitationAccumulatorNetwork;
static std::array<ConditionedReflexNeuron, CONDITIONED_REFLEX_CONTAINER_NUM*CONDITIONED_REFLEX_PER_CONTAINER> s_conditionedReflexNetwork;
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
static uint32_t s_neuronsNum = EYE_COLOR_NEURONS_NUM*3 + (uint32_t)s_motorNetwork.size() +
	(uint32_t)s_excitationAccumulatorNetwork.size() + (uint32_t)s_conditionedReflexNetwork.size()
	+ 1 // conditionedReflexCreatorNeuron
	;

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

	uint32_t step = s_neuronsNum / (uint32_t)s_threads.size();
	uint32_t remain = s_neuronsNum - step * (uint32_t)s_threads.size();
	uint32_t posBegin = 0;
	for (std::pair<uint32_t, uint32_t> &pair : s_threadNeurons)
	{
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
		&s_conditionedReflexNetwork[0], &s_conditionedReflexNetwork[0] + s_conditionedReflexNetwork.size());

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

void NervousSystem::GetStatisticsParams(int32_t &reinforcementLevelStat, int32_t &reinforcementsCountStat) const
{
	std::lock_guard<std::mutex> guard(s_statisticsMutex);
	reinforcementLevelStat = m_reinforcementLevelStat;
	reinforcementsCountStat = m_reinforcementsCountStat;
}

int32_t NervousSystem::GetReinforcementCount() const
{
	return m_reinforcementsCountStat;
}

uint64_t NervousSystem::GetTime() const
{
	return s_time;
}

void NervousSystem::NextTick(uint64_t timeOfTheUniverse)
{
	if (!s_waitThreadsCount) // Is Tick Finished
	{
		if (timeOfTheUniverse > s_time)
		{
			m_reinforcementLevelLast = m_reinforcementLevel;
			m_reinforcementLevel += s_reinforcementStorageCurrentTick;
			s_reinforcementStorageCurrentTick = 0;
			if (0 < m_reinforcementLevel)
			{
				m_reinforcementLevel -= 1 + m_reinforcementLevelSub / (300*MILLISECOND_IN_QUANTS);
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
		--s_waitThreadsCount;
		while (s_time % 2 == isTimeOdd && s_isSimulationRunning)
		{
		}
	}
}
