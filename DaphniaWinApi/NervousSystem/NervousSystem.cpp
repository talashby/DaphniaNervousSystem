#include "NervousSystem.h"
#include <array>
#include <vector>
#include <thread>
#include <atomic>
#include <assert.h>
#include "ParallelPhysics/ServerProtocol.h"
#include "ParallelPhysics/ObserverClient.h"

// constants

constexpr int32_t ReinforcementForConditionedReflex = 10; // units
constexpr int32_t ExcitationAccumulationTime = 100; // ms
constexpr uint16_t ExcitationAccumulationLimit = ExcitationAccumulationTime * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000; // units
constexpr int32_t ConditionedReflexDendritesNum = 20; // units
constexpr uint32_t ConditionedReflexLimit = 100000; // units
constexpr uint16_t SensoryNeuronReinforcementLimit = 65535; // units

//

static NervousSystem* s_nervousSystem = nullptr;
static std::vector<std::thread> s_threads;
static std::atomic<bool> s_isSimulationRunning = false;
static std::atomic<uint64_t> s_time = 0; // absolute universe time
static std::atomic<int32_t> s_waitThreadsCount = 0; // thread synchronization variable

class Neuron* GetNeuronInterface(uint32_t neuronId);

class Neuron
{
public:
	Neuron() = default;
	virtual ~Neuron() = default;

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

private:
	uint8_t m_dendrite[2]; // 0-254 - excitation 255 - connection lost
	uint8_t m_axon[2]; // 0-254 - excitation 255 - connection lost
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
	void Tick()
	{
		Neuron *neuron = GetNeuronInterface(m_dendrite);
		if (neuron->IsActive())
		{
			if (m_excitation < ExcitationAccumulationLimit)
			{
				++m_excitation;
			}
		}
		else
		{
			if (m_periodWithoutExcitation < ExcitationAccumulationLimit)
			{
				++m_periodWithoutExcitation;
			}
			if (ExcitationAccumulationLimit - m_periodWithoutExcitation > m_excitation)
			{
				m_excitation = ExcitationAccumulationLimit - m_periodWithoutExcitation;
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
	uint32_t m_dendrite[ConditionedReflexDendritesNum]; // read corresponding axon
	uint16_t m_excitation[ConditionedReflexDendritesNum]; // max: ExcitationAccumulationTime * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 quantum of time
};

std::array<std::array<SensoryNeuron, PPh::GetObserverEyeSize()>, PPh::GetObserverEyeSize()> s_eyeNetwork;
static std::array<MotorNeuron, 3> s_motorNetwork; // 0 - forward, 1 - left, 2 - right
static std::array<ExcitationAccumulatorNeuron, s_eyeNetwork[0].size()*s_eyeNetwork.size()+ s_motorNetwork.size()> s_excitationAccumulatorNetwork;
static std::array<ConditionedReflexNeuron, ConditionedReflexLimit> s_conditionedReflexNetwork;

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

void NervousSystemThread(int32_t threadNum);

void NervousSystem::Init()
{
	static_assert(ExcitationAccumulationTime * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 <= std::numeric_limits<uint16_t>::max());
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