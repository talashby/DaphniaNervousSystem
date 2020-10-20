
#include "Neurons.h"

#include "NervousSystem.h"
#include "ParallelPhysics/ObserverClient.h"
#include <assert.h>

// constants
constexpr int32_t EXCITATION_ACCUMULATION_TIME = 100; // ms
constexpr uint16_t EXCITATION_ACCUMULATION_LIMIT = EXCITATION_ACCUMULATION_TIME * MILLISECOND_IN_QUANTS; // units
constexpr uint16_t SENSORY_NEURON_REINFORCEMENT_LIMIT = 65535; // units
constexpr uint32_t SENSORY_NEURON_REINFORCEMENT_REFRESH_TIME = 5 * SECOND_IN_QUANTS;  // quantum of time
constexpr uint32_t MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME = 15 * SECOND_IN_QUANTS; // quantum of time
constexpr uint32_t MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME_DURATION = 500 * MILLISECOND_IN_QUANTS; // quantum of time
//

uint32_t Neuron::GetId()
{
	uint32_t id = NSNamespace::GetNeuronIndex(this);
	uint32_t idTypeMask = GetType() << 24;
	id |= idTypeMask;
	return id;
}

void SensoryNeuron::Init()
{
	m_reinforcementStorage = m_reinforcementStorageMax;
}

bool SensoryNeuron::IsActive() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_isActive[isTimeEven];
}

void SensoryNeuron::ExcitatorySynapse()
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	++m_dendrite[isTimeEven];
}

void SensoryNeuron::Tick()
{
	if (m_reinforcementStorage < m_reinforcementStorageMax)
	{
		if (NSNamespace::GetNSTime() - m_timeAfterReinforcement > SENSORY_NEURON_REINFORCEMENT_REFRESH_TIME)
		{
			m_reinforcementStorage = m_reinforcementStorageMax;
		}
	}

	int isTimeOdd = NSNamespace::GetNSTime() % 2;
	if (m_dendrite[isTimeOdd] > 0)
	{
		m_isActive[isTimeOdd] = true;
		m_dendrite[isTimeOdd] = 0;
		m_timeAfterReinforcement = NSNamespace::GetNSTime();
		if (m_reinforcementStorage > 0)
		{
			NSNamespace::AddReinforcement(1000);
			--m_reinforcementStorage;
		}
	}
	else
	{
		m_isActive[isTimeOdd] = false;
	}
}

void SensoryNeuronRed::Tick()
{
	SensoryNeuron::Tick();
}

void SensoryNeuronGreen::Tick()
{
	SensoryNeuron::Tick();
}

void SensoryNeuronBlue::Tick()
{
	SensoryNeuron::Tick();
}

bool SensoryNeuronBlue::IsActive() const
{
	return SensoryNeuron::IsActive();
}

void MotorNeuron::Init()
{
	assert(NSNamespace::GetNSTime()); // server should send time already
	m_spontaneusActivityTimeStart = MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME * (PPh::Rand32(100) + 50) / 100;
	m_lastExcitationTime = NSNamespace::GetNSTime();
}

void MotorNeuron::Tick()
{
	int isTimeOdd = NSNamespace::GetNSTime() % 2;
	m_accumulatedExcitation = m_accumulatedExcitation + m_dendrite[isTimeOdd];
	m_accumulatedExcitation = std::min(m_accumulatedExcitation, (uint16_t)254);
	m_axon[isTimeOdd] = static_cast<uint8_t>(m_accumulatedExcitation);
	bool isActive = false;
	if (m_accumulatedExcitation)
	{
		--m_accumulatedExcitation;
		isActive = true;
		m_lastExcitationTime = NSNamespace::GetNSTime();
	}
	if (!m_accumulatedExcitation)
	{
		if (NSNamespace::GetNSTime() - m_lastExcitationTime > m_spontaneusActivityTimeStart)
		{
			++m_accumulatedExcitation;
			m_spontaneusActivityTimeFinishAbs = NSNamespace::GetNSTime() + MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME_DURATION;
			m_spontaneusActivityTimeStart = MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME * (PPh::Rand32(67) + 66) / 100;
		}
		else if (NSNamespace::GetNSTime() < m_spontaneusActivityTimeFinishAbs)
		{
			++m_accumulatedExcitation;
		}
	}

	uint32_t index = NSNamespace::GetNeuronIndex(this);
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

void ExcitationAccumulatorNeuron::Init(uint32_t dendrite)
{
	m_dendrite = dendrite;
	m_reflexCreatorDendriteIndex = -1;
}

void ExcitationAccumulatorNeuron::Tick()
{
	Neuron *neuron = NSNamespace::GetNeuronInterface(m_dendrite);
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
	int isTimeOdd = NSNamespace::GetNSTime() % 2;
	m_accumulatedExcitationOut[isTimeOdd] = m_accumulatedExcitation;
}

uint16_t ExcitationAccumulatorNeuron::GetAccumulatedExcitation()
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_accumulatedExcitationOut[isTimeEven];
}

void ExcitationAccumulatorNeuron::SetReflexCreatorDendriteIndex(int32_t index) // -1 not connected // used by ConditionedReflexCreatorNeuron only
{
	m_reflexCreatorDendriteIndex = index;
}

int32_t ExcitationAccumulatorNeuron::GetReflexCreatorDendriteIndex() const // used by ConditionedReflexCreatorNeuron only
{
	return m_reflexCreatorDendriteIndex;
}

void ConditionedReflexCreatorNeuron::Init(ExcitationAccumulatorNeuron *begin, ExcitationAccumulatorNeuron *end, ConditionedReflexNeuron *beginCond, ConditionedReflexNeuron *endCond)
{
	static_assert(EXCITATION_ACCUMULATION_TIME * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 <= std::numeric_limits<uint16_t>::max());

	m_accumulatorBegin = begin;
	m_accumulatorCurrent = begin;
	m_accumulatorEnd = end;
	m_conditionedReflexBegin = beginCond;
	m_conditionedReflexCurrent = beginCond;
	m_conditionedReflexEnd = endCond;
	assert(m_accumulatorEnd - m_accumulatorBegin > (int32_t)m_dendrite.size());
	for (uint32_t ii = 0; ii < m_dendrite.size(); ii++)
	{
		m_dendrite[ii] = m_accumulatorCurrent->GetId();
		m_accumulatorCurrent->SetReflexCreatorDendriteIndex(ii);
		++m_accumulatorCurrent;
	}
}

ConditionedReflexDendritesArray ConditionedReflexCreatorNeuron::GetDendritesArray() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_dendriteOut[isTimeEven];
}

ConditionedReflexExitationArray ConditionedReflexCreatorNeuron::GetExcitationArray() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_excitationOut[isTimeEven];
}

void ConditionedReflexCreatorNeuron::Tick()
{
	uint32_t accumTested = 0;
	const uint32_t accumTestedMax = 10;
	uint32_t dendriteShifted = 0;
	const uint32_t dendriteShiftedMax = 100;
	bool isReinforcementGrowth = NervousSystem::Instance()->IsReinforcementGrowth();
	bool isReinforcementHappened = NervousSystem::Instance()->IsReinforcementHappened();
	while ((!isReinforcementGrowth || isReinforcementHappened) && accumTested < accumTestedMax && dendriteShifted < dendriteShiftedMax)
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
				assert(NSNamespace::GetNeuronInterface(m_dendrite[dendriteIndex]) == (Neuron*)m_accumulatorCurrent);

				auto excitationIteratorUB = std::upper_bound(m_excitation.begin(), m_excitation.end(), accumulatedExcitation);
				int32_t dendriteIndexLB = (int32_t)std::distance(m_excitation.begin(), excitationIterator);
				int32_t dendriteIndexUB = (int32_t)std::distance(m_excitation.begin(), excitationIteratorUB);
				if (dendriteIndex == dendriteIndexLB - 1 || (dendriteIndex >= dendriteIndexLB && dendriteIndex <= dendriteIndexUB) ||
					(dendriteIndex == m_excitation.size() - 1 && dendriteIndexLB == m_excitation.size()))
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
						assert(dynamic_cast<ExcitationAccumulatorNeuron*>(NSNamespace::GetNeuronInterface(m_dendrite[ii])));
						ExcitationAccumulatorNeuron* neuron = (ExcitationAccumulatorNeuron*)NSNamespace::GetNeuronInterface(m_dendrite[ii]);
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
					int32_t newDendriteIndex = (int32_t)std::distance(m_excitation.begin(), excitationIterator - 1);
					assert((uint32_t)newDendriteIndex < m_dendrite.size());
					assert(m_dendrite[0]);
					ExcitationAccumulatorNeuron* looser = (ExcitationAccumulatorNeuron*)NSNamespace::GetNeuronInterface(m_dendrite[0]);
					looser->SetReflexCreatorDendriteIndex(-1);

					for (int32_t ii = 0; ii < newDendriteIndex; ++ii)
					{
						m_dendrite[ii] = m_dendrite[ii + 1];
						m_excitation[ii] = m_excitation[ii + 1];
						assert(m_dendrite[ii]);
						assert(dynamic_cast<ExcitationAccumulatorNeuron*>(NSNamespace::GetNeuronInterface(m_dendrite[ii])));
						ExcitationAccumulatorNeuron* neuron = (ExcitationAccumulatorNeuron*)NSNamespace::GetNeuronInterface(m_dendrite[ii]);
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

	if (m_reinforcementsCount < NervousSystem::Instance()->GetReinforcementCount())
	{
		++m_reinforcementsCount;
		m_conditionedReflexCurrent->Init(m_dendrite, m_excitation);
		++m_conditionedReflexCurrent;
		assert(m_conditionedReflexCurrent < m_conditionedReflexEnd); // TMP
	}
	int isTimeOdd = NSNamespace::GetNSTime() % 2;
	m_dendriteOut[isTimeOdd] = m_dendrite;
	m_excitationOut[isTimeOdd] = m_excitation;
}

void ConditionedReflexNeuron::Init(std::array<uint32_t, CONDITIONED_REFLEX_DENDRITES_NUM> &dendrite, std::array <uint16_t, CONDITIONED_REFLEX_DENDRITES_NUM> &accumulatedExcitation)
{
	m_dendrite = dendrite;
	m_excitation = accumulatedExcitation;
	m_isActive = true;
}

void ConditionedReflexNeuron::Tick()
{
	if (m_isActive)
	{
		ConditionedReflexDendritesArray dendritesCur = NSNamespace::GetConditionedReflexCreatorNeuron()->GetDendritesArray();
		uint32_t error = 0;
		for (int ii = dendritesCur.size() - 1; ii >= 0; --ii)
		{
			uint32_t weight = (ii + 5);
			uint32_t errorCur = weight * 30;
			for (int jj = m_dendrite.size() - 1; jj >= 0; --jj)
			{
				if (dendritesCur[ii] == m_dendrite[jj])
				{
					errorCur = weight * abs(ii - jj);
				}
			}
			error += errorCur;
		}
		NervousSystem::Instance()->SetConditionedTmpStat(error);
		//ConditionedReflexExitationArray exitation = NSNamespace::GetConditionedReflexCreatorNeuron()->GetExcitationArray();
	}
	//assert(m_debugCheckTime != NSNamespace::GetNSTime()); // to check Tick calls once per quantum of time
	//m_debugCheckTime = NSNamespace::GetNSTime();
}

void ConditionedReflexContainerNeuron::Init(ConditionedReflexNeuron *begin, ConditionedReflexNeuron *end)
{
	m_conditionedReflexBegin = begin;
	m_conditionedReflexCurrent = begin;
	m_conditionedReflexEnd = end;
}

void ConditionedReflexContainerNeuron::Tick()
{
	if (NervousSystem::Instance()->IsReinforcementGrowth())
	{
		return;
	}
	m_conditionedReflexCurrent->Tick();
	m_conditionedReflexCurrent += CONDITIONED_REFLEX_PER_CONTAINER;
	if (m_conditionedReflexCurrent >= m_conditionedReflexEnd)
	{
		m_conditionedReflexCurrent = m_conditionedReflexBegin;
	}
}

void PrognosticNeuron::Init(std::array<uint32_t, CONDITIONED_REFLEX_DENDRITES_NUM> &dendrite, std::array <uint16_t, CONDITIONED_REFLEX_DENDRITES_NUM> &accumulatedExcitation)
{
	m_dendrite = dendrite;
	m_excitation = accumulatedExcitation;
}
