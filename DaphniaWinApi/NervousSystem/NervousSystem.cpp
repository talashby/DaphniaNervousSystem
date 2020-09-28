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

// constants

constexpr int32_t REINFORCEMENT_FOR_CONDITIONED_REFLEX = 10; // units
constexpr int32_t EXCITATION_ACCUMULATION_TIME = 100; // ms
constexpr uint16_t EXCITATION_ACCUMULATION_LIMIT = EXCITATION_ACCUMULATION_TIME * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000; // units
constexpr int32_t CONDITIONED_REFLEX_DENDRITES_NUM = 20; // units
constexpr uint32_t CONDITIONED_REFLEX_LIMIT = 100000; // units
constexpr uint16_t SENSORY_NEURON_REINFORCEMENT_LIMIT = 65535; // units

//

static NervousSystem* s_nervousSystem = nullptr;
static std::array<std::thread, 1> s_threads;
static std::array<std::pair<uint32_t, uint32_t>, s_threads.size()> s_threadNeurons; // [first neuron; last neuron)
static std::atomic<bool> s_isSimulationRunning = false;
static std::atomic<uint64_t> s_time = 0; // absolute universe time
static std::atomic<int32_t> s_waitThreadsCount = 0; // thread synchronization variable

class Neuron* GetNeuronInterface(uint32_t neuronId);
uint32_t GetNeuronIndex(Neuron *neuron);

class Neuron
{
public:
	Neuron() = default;
	virtual ~Neuron() = default;

	virtual void Tick() {}

	virtual bool IsActive()
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

	bool IsActive() override
	{
		int isTimeOdd = (s_time + 1) % 2;
		return m_dendrite[isTimeOdd] > 0;
	}
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

private:
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
		}

		uint32_t index = GetNeuronIndex(this);
		switch (index)
		{
		case 0:
			PPh::ObserverClient::Instance()->SetIsForward(isActive);
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
};

class ExcitationAccumulatorNeuron : public Neuron
{
public:
	ExcitationAccumulatorNeuron() = default;
	virtual ~ExcitationAccumulatorNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::ExcitationAccumulatorNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

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
	uint32_t m_firstNeuronNum;
	uint32_t m_lastNeuronNum;
	Neuron *m_start;
	Neuron *m_end;
	uint32_t m_size;
};
static std::array s_networksMetadata{
	NetworksMetadata{0, 0, &s_eyeNetwork[0][0], &s_eyeNetwork[0][0]+EYE_NEURONS_NUM, sizeof(SensoryNeuron)},
	NetworksMetadata{0, 0, &s_motorNetwork[0], &s_motorNetwork[0]+s_motorNetwork.size(), sizeof(MotorNeuron)},
	NetworksMetadata{0, 0, &s_excitationAccumulatorNetwork[0], &s_excitationAccumulatorNetwork[0] + s_excitationAccumulatorNetwork.size(), sizeof(ExcitationAccumulatorNeuron)},
	NetworksMetadata{0, 0, &s_conditionedReflexNetwork[0], &s_conditionedReflexNetwork[0] + s_conditionedReflexNetwork.size(), sizeof(ConditionedReflexNeuron)}
};
static uint32_t s_neuronsNum = EYE_NEURONS_NUM + s_motorNetwork.size() + s_excitationAccumulatorNetwork.size() + s_conditionedReflexNetwork.size();

class Neuron* GetNeuronInterface(uint32_t neuronId)
{
	Neuron* neuron = nullptr;
	uint8_t neuronType = (neuronId >> 24) & 0xff;
	uint16_t neuronIndex = neuronId & 0xffff;
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
	assert(((int32_t*)neuron)[0] > 0); // virtual table should exist
	return neuron;
}

uint32_t GetNeuronIndex(Neuron *neuron)
{
	for (auto &metadata : s_networksMetadata)
	{
		if (neuron >= metadata.m_start && neuron < metadata.m_end)
		{
			return (neuron - metadata.m_start) / metadata.m_size;
		}
	}
	assert(false);
	return 0;
}

void NervousSystemThread(int32_t threadNum);

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
		el.m_firstNeuronNum = firstNeuronNum;
		el.m_lastNeuronNum = firstNeuronNum + (reinterpret_cast<uint64_t>(el.m_end) - reinterpret_cast<uint64_t>(el.m_start)) / el.m_size;
		firstNeuronNum = el.m_lastNeuronNum;
	}
	

	uint32_t step = s_neuronsNum / s_threads.size();
	uint32_t remain = s_neuronsNum - step * s_threads.size();
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