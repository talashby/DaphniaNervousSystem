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
#include <algorithm>
#include <xutility>

// constants

constexpr uint32_t SECOND_IN_QOF = PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND;  // quantum of time
constexpr uint32_t MILLISECOND_IN_QOF = PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000;  // quantum of time
constexpr int32_t REINFORCEMENT_FOR_CONDITIONED_REFLEX = 20000; // units
constexpr int32_t EXCITATION_ACCUMULATION_TIME = 100; // ms
constexpr uint16_t EXCITATION_ACCUMULATION_LIMIT = EXCITATION_ACCUMULATION_TIME * MILLISECOND_IN_QOF; // units
constexpr int32_t CONDITIONED_REFLEX_DENDRITES_NUM = 20; // units
constexpr uint32_t CONDITIONED_REFLEX_LIMIT = 100; // units
constexpr uint16_t SENSORY_NEURON_REINFORCEMENT_LIMIT = 65535; // units
constexpr uint32_t SENSORY_NEURON_REINFORCEMENT_REFRESH_TIME = 10 * SECOND_IN_QOF;  // quantum of time
constexpr uint32_t MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME = 10 * SECOND_IN_QOF; // quantum of time
constexpr uint32_t MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME_DURATION = 500 * MILLISECOND_IN_QOF; // quantum of time
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
	uint32_t GetId()
	{
		uint32_t id = GetNeuronIndex(this);
		uint32_t idTypeMask = GetType() << 24;
		id |= idTypeMask;
		return id;
	}

protected:
	enum class NeuronTypes
	{
		None = 0,
		SensoryNeuron,
		SensoryNeuronRed,
		SensoryNeuronGreen,
		SensoryNeuronBlue,
		MotorNeuron,
		ExcitationAccumulatorNeuron,
		ConditionedReflexCreatorNeuron,
		ConditionedReflexNeuron,
		ConditionedReflexContainerNeuron
	};
};

class SensoryNeuron : public Neuron
{
public:
	SensoryNeuron() = default;
	virtual ~SensoryNeuron() = default;

	void Init()  override 
	{
		m_reinforcementStorage = m_reinforcementStorageMax;
	}

	bool IsActive() const override
	{
		int isTimeEven = (s_time + 1) % 2;
		return m_isActive[isTimeEven];
	}
	void ExcitatorySynapse()
	{
		int isTimeEven = (s_time+1) % 2;
		++m_dendrite[isTimeEven];
	}
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override
	{
		if (m_reinforcementStorage < m_reinforcementStorageMax)
		{
			if (s_time - m_timeAfterReinforcement > SENSORY_NEURON_REINFORCEMENT_REFRESH_TIME)
			{
				m_reinforcementStorage = m_reinforcementStorageMax;
			}
		}

		int isTimeOdd = s_time % 2;
		if (m_dendrite[isTimeOdd] > 0)
		{
			m_isActive[isTimeOdd] = true;
			m_dendrite[isTimeOdd] = 0;
			m_timeAfterReinforcement = s_time;
			if (m_reinforcementStorage > 0)
			{
				s_reinforcementStorageCurrentTick.fetch_add(1000, std::memory_order_relaxed);
				--m_reinforcementStorage;
			}
		}
		else
		{
			m_isActive[isTimeOdd] = false;
		}
	}

private:
	static const uint32_t m_reinforcementStorageMax = 5;
	bool m_isActive[2];
	uint8_t m_dendrite[2]; // 0-254 - excitation 255 - connection lost
	uint8_t m_axon[2]; // 0-254 - excitation 255 - connection lost
	bool m_isReinforcementActive[2]; // 
	uint32_t m_reinforcementCounter; // how many times reinforcement happened. Stop refresh when SensoryNeuronReinforcementLimit will be reached
	uint64_t m_timeAfterReinforcement; // time counter to refresh reinforcement
	uint32_t m_reinforcementStorage;
};

class SensoryNeuronRed : public SensoryNeuron
{
public:
	SensoryNeuronRed() = default;
	virtual ~SensoryNeuronRed() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuronRed); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override
	{
		SensoryNeuron::Tick();
	}
};

class SensoryNeuronGreen : public SensoryNeuron
{
public:
	SensoryNeuronGreen() = default;
	virtual ~SensoryNeuronGreen() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuronGreen); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override
	{
		SensoryNeuron::Tick();
	}
};

class SensoryNeuronBlue : public SensoryNeuron
{
public:
	SensoryNeuronBlue() = default;
	virtual ~SensoryNeuronBlue() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuronBlue); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override
	{
		SensoryNeuron::Tick();
	}
	bool IsActive() const override
	{
		return SensoryNeuron::IsActive();
	}
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
		m_accumulatedExcitation = m_accumulatedExcitation + m_dendrite[isTimeOdd];
		m_accumulatedExcitation = std::min(m_accumulatedExcitation, (uint16_t)254);
		m_axon[isTimeOdd] = static_cast<uint8_t>(m_accumulatedExcitation);
		bool isActive = false;
		if (m_accumulatedExcitation)
		{
			--m_accumulatedExcitation;
			isActive = true;
			m_lastExcitationTime = s_time;
		}
		if (!m_accumulatedExcitation)
		{
			if (s_time - m_lastExcitationTime > m_spontaneusActivityTimeStart)
			{
				++m_accumulatedExcitation;
				m_spontaneusActivityTimeFinishAbs = s_time + MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME_DURATION;
				m_spontaneusActivityTimeStart = MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME * (PPh::Rand32(67) + 66) / 100;
			}
			else if (s_time < m_spontaneusActivityTimeFinishAbs)
			{
				++m_accumulatedExcitation;
			}
		}

		uint32_t index = GetNeuronIndex(this);
		switch (index)
		{
		case 0:
			//PPh::ObserverClient::Instance()->SetIsForward(isActive);
			break;
		case 1:
			//PPh::ObserverClient::Instance()->SetIsLeft(isActive);
			break;
		case 2:
			//PPh::ObserverClient::Instance()->SetIsRight(isActive);
			break;
		}
	}

private:
	uint8_t m_dendrite[2]; // 0-254 - excitation 255 - connection lost
	uint8_t m_axon[2]; // 0-254 - excitation 255 - connection lost
	uint16_t m_accumulatedExcitation;
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
		m_reflexCreatorDendriteIndex = -1;
	}

	void Tick() override
	{
		Neuron *neuron = GetNeuronInterface(m_dendrite);
		if (neuron->IsActive())
		{
			if (m_accumulatedExcitation < EXCITATION_ACCUMULATION_LIMIT)
			{
				++m_accumulatedExcitation;
				m_periodWithoutExcitation = 0;
			}
		}
		else
		{
			if (m_periodWithoutExcitation < EXCITATION_ACCUMULATION_LIMIT)
			{
				++m_periodWithoutExcitation;
			}
			if (EXCITATION_ACCUMULATION_LIMIT - m_periodWithoutExcitation < m_accumulatedExcitation)
			{
				m_accumulatedExcitation = EXCITATION_ACCUMULATION_LIMIT - m_periodWithoutExcitation;
			}
		}
		int isTimeOdd = s_time % 2;
		m_accumulatedExcitationOut[isTimeOdd] = m_accumulatedExcitation;
	}

	uint16_t GetAccumulatedExcitation()
	{
		int isTimeEven = (s_time+1) % 2;
		return m_accumulatedExcitationOut[isTimeEven];
	}
	
	void SetReflexCreatorDendriteIndex(int32_t index) // -1 not connected // used by ConditionedReflexCreatorNeuron only
	{
		m_reflexCreatorDendriteIndex = index;
	}

	int32_t GetReflexCreatorDendriteIndex() const // used by ConditionedReflexCreatorNeuron only
	{
		return m_reflexCreatorDendriteIndex;
	}
private:
	uint32_t m_dendrite; // read corresponding axon
	uint16_t m_accumulatedExcitation; // max: ExcitationAccumulationTime * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 quantum of time
	uint16_t m_accumulatedExcitationOut[2];
	uint16_t m_periodWithoutExcitation; // quantum of time
	int32_t m_reflexCreatorDendriteIndex; // used by ConditionedReflexCreatorNeuron only
};

class ConditionedReflexCreatorNeuron : public Neuron
{
public:
	ConditionedReflexCreatorNeuron() = default;
	virtual ~ConditionedReflexCreatorNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::ConditionedReflexCreatorNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }
	void Init(ExcitationAccumulatorNeuron *begin, ExcitationAccumulatorNeuron *end)
	{
		m_accumulatorBegin = begin;
		m_accumulatorCurrent = begin;
		m_accumulatorEnd = end;
		assert(m_accumulatorEnd - m_accumulatorBegin > (int32_t)m_dendrite.size());
		for (uint32_t ii = 0; ii < m_dendrite.size(); ii++)
		{
			m_dendrite[ii] = m_accumulatorCurrent->GetId();
			m_accumulatorCurrent->SetReflexCreatorDendriteIndex(ii);
			++m_accumulatorCurrent;
		}
	}

	/*bool CheckPriority()
	{
		for (uint32_t ii=0; ii<m_excitation.size()-1; ++ii)
		{
			if (m_excitation[ii] > m_excitation[ii + 1])
			{
				return false;
			}
		}
		return true;
	}*/

	void Tick() override
	{
		uint32_t accumTested = 0;
		const uint32_t accumTestedMax = 10;
		uint32_t dendriteShifted = 0;
		const uint32_t dendriteShiftedMax = 100;
		while (accumTested < accumTestedMax && dendriteShifted < dendriteShiftedMax)
		{
			++accumTested;
			uint32_t isExist = ((uint32_t*)m_accumulatorCurrent)[0]; // check virtual methods table
			if (isExist)
			{
				uint16_t accumulatedExcitation = m_accumulatorCurrent->GetAccumulatedExcitation();
				auto excitationIterator = std::lower_bound(m_excitation.begin(), m_excitation.end(), accumulatedExcitation);
				int32_t dendriteIndex = m_accumulatorCurrent->GetReflexCreatorDendriteIndex();
				if (dendriteIndex != -1)
				{
					assert(GetNeuronInterface(m_dendrite[dendriteIndex]) == (Neuron*)m_accumulatorCurrent);

					auto excitationIteratorUB = std::upper_bound(m_excitation.begin(), m_excitation.end(), accumulatedExcitation);
					int32_t dendriteIndexLB = (int32_t)std::distance(m_excitation.begin(), excitationIterator);
					int32_t dendriteIndexUB = (int32_t)std::distance(m_excitation.begin(), excitationIteratorUB);
					if (dendriteIndex == dendriteIndexLB-1 || (dendriteIndex >= dendriteIndexLB && dendriteIndex <= dendriteIndexUB) ||
						(dendriteIndex==m_excitation.size()-1 && dendriteIndexLB == m_excitation.size()))
					{
						m_excitation[dendriteIndex] = accumulatedExcitation;
					}
					else
					{
						int32_t newDendriteIndex = dendriteIndexLB; //
						if (dendriteIndex < newDendriteIndex)
						{
							--newDendriteIndex;
						}
						assert(dendriteIndex != newDendriteIndex);
						int sign = PPh::Sign(newDendriteIndex - dendriteIndex);
						for (int32_t ii = dendriteIndex; ii != newDendriteIndex; ii += sign)
						{
							m_dendrite[ii] = m_dendrite[ii + sign];
							m_excitation[ii] = m_excitation[ii + sign];
							assert(dynamic_cast<ExcitationAccumulatorNeuron*>(GetNeuronInterface(m_dendrite[ii])));
							ExcitationAccumulatorNeuron* neuron = (ExcitationAccumulatorNeuron*)GetNeuronInterface(m_dendrite[ii]);
							assert(ii + sign == neuron->GetReflexCreatorDendriteIndex());
							neuron->SetReflexCreatorDendriteIndex(ii);
						}
						m_dendrite[newDendriteIndex] = m_accumulatorCurrent->GetId();
						m_excitation[newDendriteIndex] = accumulatedExcitation;
						m_accumulatorCurrent->SetReflexCreatorDendriteIndex(newDendriteIndex);
						dendriteShifted += abs(newDendriteIndex - dendriteIndex);
					}
				}
				else
				{
					if (excitationIterator != m_excitation.begin())
					{
						int32_t newDendriteIndex = (int32_t)std::distance(m_excitation.begin(), excitationIterator-1);
						assert((uint32_t)newDendriteIndex < m_dendrite.size());
						assert(m_dendrite[0]);
						ExcitationAccumulatorNeuron* looser = (ExcitationAccumulatorNeuron*)GetNeuronInterface(m_dendrite[0]);
						looser->SetReflexCreatorDendriteIndex(-1);

						for (int32_t ii = 0; ii < newDendriteIndex; ++ii)
						{
							m_dendrite[ii] = m_dendrite[ii + 1];
							m_excitation[ii] = m_excitation[ii + 1];
							assert(m_dendrite[ii]);
							assert(dynamic_cast<ExcitationAccumulatorNeuron*>(GetNeuronInterface(m_dendrite[ii])));
							ExcitationAccumulatorNeuron* neuron = (ExcitationAccumulatorNeuron*)GetNeuronInterface(m_dendrite[ii]);
							assert(ii + 1 == neuron->GetReflexCreatorDendriteIndex());
							neuron->SetReflexCreatorDendriteIndex(ii);
						}
						m_dendrite[newDendriteIndex] = m_accumulatorCurrent->GetId();
						m_excitation[newDendriteIndex] = accumulatedExcitation;
						m_accumulatorCurrent->SetReflexCreatorDendriteIndex(newDendriteIndex);
						dendriteShifted += newDendriteIndex;
					}
				}
			}
			++m_accumulatorCurrent;
			if (m_accumulatorCurrent >= m_accumulatorEnd)
			{
				m_accumulatorCurrent = m_accumulatorBegin;
			}
		}
	}

private:
	std::array<uint32_t, CONDITIONED_REFLEX_DENDRITES_NUM> m_dendrite; // read corresponding axon
	std::array <uint16_t, CONDITIONED_REFLEX_DENDRITES_NUM> m_excitation; // max: ExcitationAccumulationTime * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 quantum of time
	ExcitationAccumulatorNeuron *m_accumulatorBegin;
	ExcitationAccumulatorNeuron *m_accumulatorEnd;
	ExcitationAccumulatorNeuron *m_accumulatorCurrent;
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
	uint16_t m_accumulatedExcitation[CONDITIONED_REFLEX_DENDRITES_NUM]; // max: ExcitationAccumulationTime * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 quantum of time
};

class ConditionedReflexContainerNeuron : public Neuron
{
public:
	ConditionedReflexContainerNeuron() = default;
	virtual ~ConditionedReflexContainerNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::ConditionedReflexContainerNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Init(ConditionedReflexNeuron *begin, ConditionedReflexNeuron *end)
	{
		m_conditionedReflexBegin = begin;
		m_conditionedReflexCurrent = begin;
		m_conditionedReflexEnd = end;
	}
	void Tick() override
	{
		m_conditionedReflexCurrent->Tick();
		++m_conditionedReflexCurrent;
		if (m_conditionedReflexCurrent < m_conditionedReflexEnd)
		{
			m_conditionedReflexCurrent = m_conditionedReflexBegin;
		}
	}
private:
	ConditionedReflexNeuron *m_conditionedReflexBegin;
	ConditionedReflexNeuron *m_conditionedReflexEnd;
	ConditionedReflexNeuron *m_conditionedReflexCurrent;
};

constexpr static int EYE_COLOR_NEURONS_NUM =  PPh::GetObserverEyeSize()*PPh::GetObserverEyeSize();
static std::array<std::array<SensoryNeuronRed, PPh::GetObserverEyeSize()>, PPh::GetObserverEyeSize()> s_eyeNetworkRed;
static std::array<std::array<SensoryNeuronGreen, PPh::GetObserverEyeSize()>, PPh::GetObserverEyeSize()> s_eyeNetworkGreen;
static std::array<std::array<SensoryNeuronBlue, PPh::GetObserverEyeSize()>, PPh::GetObserverEyeSize()> s_eyeNetworkBlue;
static std::array<MotorNeuron, 3> s_motorNetwork; // 0 - forward, 1 - left, 2 - right
static std::array<ExcitationAccumulatorNeuron, EYE_COLOR_NEURONS_NUM*3 + s_motorNetwork.size()> s_excitationAccumulatorNetwork;
static std::array<ConditionedReflexNeuron, CONDITIONED_REFLEX_LIMIT> s_conditionedReflexNetwork;
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
	NetworksMetadata{0, 0, (uint64_t)&s_conditionedReflexNetwork[0], (uint64_t)(&s_conditionedReflexNetwork[0] + s_conditionedReflexNetwork.size()), sizeof(ConditionedReflexNeuron)},
	NetworksMetadata{0, 0, (uint64_t)&s_conditionedReflexCreatorNeuron, (uint64_t)(&s_conditionedReflexCreatorNeuron + 1), sizeof(ConditionedReflexCreatorNeuron)}
};
static uint32_t s_neuronsNum = EYE_COLOR_NEURONS_NUM*3 + (uint32_t)s_motorNetwork.size() +
	(uint32_t)s_excitationAccumulatorNetwork.size() + (uint32_t)s_conditionedReflexNetwork.size()
	+ 1 // conditionedReflexCreatorNeuron
	;

class Neuron* GetNeuronInterface(uint32_t neuronId)
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
	assert(s_excitationAccumulatorNetwork.size() == excitationAccumulatorNetworkIndex);
	s_conditionedReflexCreatorNeuron.Init(&s_excitationAccumulatorNetwork[0], &s_excitationAccumulatorNetwork[s_excitationAccumulatorNetwork.size() - 1]);

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
				m_reinforcementLevel -= 1 + m_reinforcementLevelSub / (300*MILLISECOND_IN_QOF);
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