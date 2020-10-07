#include "NervousSystem.h"
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

// constants

constexpr int32_t REINFORCEMENT_FOR_CONDITIONED_REFLEX = 10; // units
constexpr int32_t EXCITATION_ACCUMULATION_TIME = 100; // ms
constexpr uint16_t EXCITATION_ACCUMULATION_LIMIT = EXCITATION_ACCUMULATION_TIME * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000; // units
constexpr int32_t CONDITIONED_REFLEX_DENDRITES_NUM = 20; // units
constexpr uint32_t CONDITIONED_REFLEX_LIMIT = 10000; // units
constexpr uint16_t SENSORY_NEURON_REINFORCEMENT_LIMIT = 65535; // units
constexpr uint32_t MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME = 10000 * // ms
					PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000; // quantum of time
constexpr uint32_t MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME_DURATION = 500 * // ms
					PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000; // quantum of time
//

static NervousSystem* s_nervousSystem = nullptr;
static std::array<std::thread, 3> s_threads;
static std::array<std::pair<uint32_t, uint32_t>, s_threads.size()> s_threadNeurons; // [first neuron; last neuron)
static std::atomic<bool> s_isSimulationRunning = false;
static std::atomic<uint64_t> s_time = 0; // absolute universe time
static std::atomic<uint32_t> s_waitThreadsCount = 0; // thread synchronization variable
static std::atomic<uint32_t> s_reinforcementStorageCurrentTick = 0; // store reinforcements in current tick

static std::mutex s_statisticsMutex;

class Neuron* GetNeuronInterface(uint32_t neuronId);
uint32_t GetNeuronIndex(Neuron *neuron);

class Neuron
{
public:
	Neuron() = default;
	virtual ~Neuron() = default;

	virtual void Init() {}
	virtual void Tick() {}

	virtual bool IsActive() const
	{
		return false;
	}

	virtual uint8_t GetType() = 0;

protected:
	enum class NeuronTypes
	{
		None = 0,
		SensoryNeuron,
		MotorNeuron,
		ExcitationAccumulatorNeuron,
		ConditionedReflexNeuron
	};
};

class SensoryNeuron : public Neuron
{
public:
	SensoryNeuron() = default;
	virtual ~SensoryNeuron() = default;

	bool IsActive() const override
	{
		return m_isActive;
	}
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override
	{
		int isTimeOdd = s_time % 2;
		if (m_dendrite[isTimeOdd] > 0)
		{
			m_isActive = true;
			m_dendrite[isTimeOdd] = 0;
			++s_reinforcementStorageCurrentTick;
		}
		
	}

private:
	bool m_isActive;
	uint8_t m_dendrite[2]; // 0-254 - excitation 255 - connection lost
	uint8_t m_axon[2]; // 0-254 - excitation 255 - connection lost
	bool m_isReinforcementActive[2]; // 
	uint32_t m_reinforcementCounter; // how many times reinforcement happened. Stop refresh when SensoryNeuronReinforcementLimit will be reached
	uint32_t m_timeAfterReinforcement; // time counter to refresh reinforcement
};

class MotorNeuron : public Neuron
{
public:
	MotorNeuron() = default;
	virtual ~MotorNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::MotorNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }
	void Init() override 
	{
		assert(s_time); // server should send time already
		m_spontaneusActivityTimeStart = MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME * (PPh::Rand32(100) + 50) / 100;
		m_lastExcitationTime = s_time;
	}

	void Tick() override
	{
		int isTimeOdd = s_time % 2;
		m_excitation = m_excitation + m_dendrite[isTimeOdd];
		m_excitation = std::min(m_excitation, (uint16_t)254);
		m_axon[isTimeOdd] = static_cast<uint8_t>(m_excitation);
		bool isActive = false;
		if (m_excitation)
		{
			--m_excitation;
			isActive = true;
			m_lastExcitationTime = s_time;
		}
		if (!m_excitation)
		{
			if (s_time - m_lastExcitationTime > m_spontaneusActivityTimeStart)
			{
				++m_excitation;
				m_spontaneusActivityTimeFinishAbs = s_time + MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME_DURATION;
				m_spontaneusActivityTimeStart = MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME * (PPh::Rand32(67) + 66) / 100;
			}
			else if (s_time < m_spontaneusActivityTimeFinishAbs)
			{
				++m_excitation;
			}
		}

		uint32_t index = GetNeuronIndex(this);
		switch (index)
		{
		case 0:
			//PPh::ObserverClient::Instance()->SetIsForward(isActive);
			break;
		case 1:
			PPh::ObserverClient::Instance()->SetIsLeft(isActive);
			break;
		case 2:
			PPh::ObserverClient::Instance()->SetIsRight(isActive);
			break;
		}
	}

private:
	uint8_t m_dendrite[2]; // 0-254 - excitation 255 - connection lost
	uint8_t m_axon[2]; // 0-254 - excitation 255 - connection lost
	uint16_t m_excitation;
	uint64_t m_lastExcitationTime;
	uint32_t m_spontaneusActivityTimeStart;
	uint64_t m_spontaneusActivityTimeFinishAbs;
};

class ExcitationAccumulatorNeuron : public Neuron
{
public:
	ExcitationAccumulatorNeuron() = default;
	virtual ~ExcitationAccumulatorNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::ExcitationAccumulatorNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }
	void Init(uint32_t dendrite)
	{
		m_dendrite = dendrite;
	}

private:
	uint32_t m_dendrite; // read corresponding axon
	uint16_t m_excitation; // max: ExcitationAccumulationTime * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 quantum of time
	uint16_t m_periodWithoutExcitation; // quantum of time
	void Tick() override
	{
		Neuron *neuron = GetNeuronInterface(m_dendrite);
		if (neuron->IsActive())
		{
			if (m_excitation < EXCITATION_ACCUMULATION_LIMIT)
			{
				++m_excitation;
			}
		}
		else
		{
			if (m_periodWithoutExcitation < EXCITATION_ACCUMULATION_LIMIT)
			{
				++m_periodWithoutExcitation;
			}
			if (EXCITATION_ACCUMULATION_LIMIT - m_periodWithoutExcitation > m_excitation)
			{
				m_excitation = EXCITATION_ACCUMULATION_LIMIT - m_periodWithoutExcitation;
			}
		}
	}
};

class ConditionedReflexNeuron : public Neuron
{
public:
	ConditionedReflexNeuron() = default;
	virtual ~ConditionedReflexNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::ConditionedReflexNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

private:
	uint32_t m_dendrite[CONDITIONED_REFLEX_DENDRITES_NUM]; // read corresponding axon
	uint16_t m_excitation[CONDITIONED_REFLEX_DENDRITES_NUM]; // max: ExcitationAccumulationTime * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 quantum of time
};

constexpr static int EYE_NEURONS_NUM =  PPh::GetObserverEyeSize()*PPh::GetObserverEyeSize();
static std::array<std::array<SensoryNeuron, PPh::GetObserverEyeSize()>, PPh::GetObserverEyeSize()> s_eyeNetwork;
static std::array<MotorNeuron, 3> s_motorNetwork; // 0 - forward, 1 - left, 2 - right
static std::array<ExcitationAccumulatorNeuron, EYE_NEURONS_NUM + s_motorNetwork.size()> s_excitationAccumulatorNetwork;
static std::array<ConditionedReflexNeuron, CONDITIONED_REFLEX_LIMIT> s_conditionedReflexNetwork;
struct NetworksMetadata
{
	uint32_t m_beginNeuronNum;
	uint32_t m_endNeuronNum; // last+1
	uint64_t m_begin;
	uint64_t m_end;
	uint32_t m_size;
};
static std::array s_networksMetadata{
	NetworksMetadata{0, 0, (uint64_t)(&s_eyeNetwork[0][0]), (uint64_t)(&s_eyeNetwork[0][0]+EYE_NEURONS_NUM), sizeof(SensoryNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_motorNetwork[0], (uint64_t)(&s_motorNetwork[0]+s_motorNetwork.size()), sizeof(MotorNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_excitationAccumulatorNetwork[0], (uint64_t)(&s_excitationAccumulatorNetwork[0] + s_excitationAccumulatorNetwork.size()), sizeof(ExcitationAccumulatorNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_conditionedReflexNetwork[0], (uint64_t)(&s_conditionedReflexNetwork[0] + s_conditionedReflexNetwork.size()), sizeof(ConditionedReflexNeuron)}
};
static uint32_t s_neuronsNum = EYE_NEURONS_NUM + (uint32_t)s_motorNetwork.size() + (uint32_t)s_excitationAccumulatorNetwork.size() + (uint32_t)s_conditionedReflexNetwork.size();

class Neuron* GetNeuronInterface(uint32_t neuronId)
{
	Neuron* neuron = nullptr;
	uint8_t neuronType = (neuronId >> 24) & 0xff;
	uint16_t neuronIndex = neuronId & 0xffffff;
	switch (neuronType)
	{
	case SensoryNeuron::GetTypeStatic():
	{
		assert(s_eyeNetwork.size());
		assert(neuronIndex < s_eyeNetwork.size()*s_eyeNetwork[0].size());
		uint16_t xx = neuronIndex / PPh::GetObserverEyeSize();
		uint16_t yy = neuronIndex - xx * PPh::GetObserverEyeSize();
		neuron = &s_eyeNetwork[xx][yy];
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
	}
	assert(((uint64_t*)neuron)[0] > 0); // virtual table should exist
	return neuron;
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

void NervousSystemThread(uint32_t threadNum);

void NervousSystem::Init()
{
	static_assert(EXCITATION_ACCUMULATION_TIME * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 <= std::numeric_limits<uint16_t>::max());
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
	for (uint32_t ii=0; ii<EYE_NEURONS_NUM; ++ii)
	{
		uint32_t id = ii;
		uint32_t type = SensoryNeuron::GetTypeStatic();
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
				--m_reinforcementLevel;
				std::lock_guard<std::mutex> guard(s_statisticsMutex);
				m_reinforcementLevelStat = m_reinforcementLevel;
				if (m_reinforcementLevel >= REINFORCEMENT_FOR_CONDITIONED_REFLEX && m_reinforcementLevelLast < REINFORCEMENT_FOR_CONDITIONED_REFLEX)
				{
					++m_reinforcementsCountStat;
				}
			}
			s_waitThreadsCount = (uint32_t)s_threads.size();
			++s_time;
		}
	}
}

void NervousSystem::PhotonReceived(uint8_t m_posX, uint8_t m_posY, PPh::EtherColor m_color)
{
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
	uint32_t beginLocalNeuronNum = s_threadNeurons[threadNum].first;
	uint32_t endLocalNeuronNum = s_threadNeurons[threadNum].second;
	while (s_isSimulationRunning)
	{
		int32_t isTimeOdd = s_time % 2;

		for (uint32_t ii = beginNetworkMetadata; ii < endNetworkMetadata; ++ii)
		{
			if (ii != beginNetworkMetadata)
			{
				beginLocalNeuronNum = s_networksMetadata[ii].m_beginNeuronNum;
			}
			for (uint32_t jj = beginLocalNeuronNum; jj < endLocalNeuronNum && jj < s_networksMetadata[ii].m_endNeuronNum; ++jj)
			{
				Neuron *neuron = reinterpret_cast<Neuron*>(s_networksMetadata[ii].m_begin + (jj - beginLocalNeuronNum) * s_networksMetadata[ii].m_size);
				neuron->Tick();
			}
		}
		--s_waitThreadsCount;
		while (s_time % 2 == isTimeOdd && s_isSimulationRunning)
		{
		}
	}
}