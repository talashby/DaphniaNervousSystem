
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

bool Neuron::IsActive() const
{
	assert(false);
	return false;
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

bool SensoryNeuron::IsReinforcementActive() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_isReinforcementActive[isTimeEven];
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
	m_isReinforcementActive[isTimeOdd] = m_reinforcementStorage > 0;
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

bool MotorNeuron::IsActive() const
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	return m_isActive[isTimeEven];
}

void MotorNeuron::Tick()
{
	int isTimeOdd = NSNamespace::GetNSTime() % 2;
	m_accumulatedExcitation = m_accumulatedExcitation + m_dendrite[isTimeOdd];
	m_dendrite[isTimeOdd] = 0;
	m_accumulatedExcitation = std::min(m_accumulatedExcitation, (uint16_t)254);
	m_axon[isTimeOdd] = static_cast<uint8_t>(m_accumulatedExcitation);
	bool isActive = false;
	if (m_accumulatedExcitation)
	{
		--m_accumulatedExcitation;
		isActive = true;
		m_lastExcitationTime = NSNamespace::GetNSTime();
	}
	if (!m_accumulatedExcitation && !NSNamespace::GetConditionedReflexCreatorNeuron()->GetConditionedReflexProceed())
	{
		if (NSNamespace::GetNSTime() - m_lastExcitationTime > m_spontaneusActivityTimeStart)
		{
			++m_accumulatedExcitation;
			m_spontaneusActivityTimeFinishAbs = NSNamespace::GetNSTime() + MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME_DURATION;
			m_spontaneusActivityTimeStart = MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME * (PPh::Rand32(67) + 66) / 100;
			NervousSystem::Instance()->SetStatus(NervousSystem::NervousSystemStatus::SpontaneousActivity);
		}
		else if (NSNamespace::GetNSTime() < m_spontaneusActivityTimeFinishAbs)
		{
			++m_accumulatedExcitation;
			NervousSystem::Instance()->SetStatus(NervousSystem::NervousSystemStatus::SpontaneousActivity);
		}
		else if (NSNamespace::GetNSTime() == m_spontaneusActivityTimeFinishAbs)
		{
			NervousSystem::Instance()->SetStatus(NervousSystem::NervousSystemStatus::Relaxing);
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
	m_isActive[isTimeOdd] = isActive;
}

void MotorNeuron::ExcitatorySynapse()
{
	int isTimeEven = (NSNamespace::GetNSTime() + 1) % 2;
	++m_dendrite[isTimeEven];
}

void ExcitationAccumulatorNeuron::Init(uint32_t dendrite)
{
	m_dendrite = dendrite;
	m_reflexCreatorDendriteIndex = -1;
}

bool ExcitationAccumulatorNeuron::IsMotorNeuron() const
{
	Neuron *neuron = NSNamespace::GetNeuronInterface(m_dendrite);
	return neuron->GetType() == MotorNeuron::GetTypeStatic();
}

bool ExcitationAccumulatorNeuron::IsReinforcementActive() const
{
	Neuron *neuron = NSNamespace::GetNeuronInterface(m_dendrite);
	return neuron->IsReinforcementActive();
}

uint32_t ExcitationAccumulatorNeuron::GetDendriteNeuron() const
{
	return m_dendrite;
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

void ConditionedReflexCreatorNeuron::Init(ExcitationAccumulatorNeuron *begin, ExcitationAccumulatorNeuron *end, ConditionedReflexNeuron *beginCond, ConditionedReflexNeuron *endCond,
	PrognosticNeuron *beginPrognostic, PrognosticNeuron *endPrognostic)
{
	static_assert(EXCITATION_ACCUMULATION_TIME * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 <= std::numeric_limits<uint16_t>::max());

	m_accumulatorBegin = begin;
	m_accumulatorCurrent = begin;
	m_accumulatorEnd = end;
	m_conditionedReflexBegin = beginCond;
	m_conditionedReflexCurrent = beginCond;
	m_conditionedReflexEnd = endCond;
	m_prognosticBegin = beginPrognostic;
	m_prognosticCurrent = beginPrognostic;
	m_prognosticEnd = endPrognostic;
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

void ConditionedReflexCreatorNeuron::AccumulateExcitation(bool isFullCircle)
{
	uint32_t accumTested = 0;
	uint32_t accumTestedMax = 200;
//	uint32_t dendriteShifted = 0;
//	uint32_t dendriteShiftedMax = 20000;
	bool isReinforcementLow = NervousSystem::Instance()->IsReinforcementLevelLow();
	bool isReinforcementHappened = NervousSystem::Instance()->IsReinforcementHappened();

	if (isFullCircle) // used to create prognostic neuron
	{
		accumTestedMax = NSNamespace::GetAccumulatorExitationNeuronNum();
		//dendriteShiftedMax = -1;
		isReinforcementLow = true;
		isReinforcementHappened = true;
	}
	

	while ((isReinforcementLow || isReinforcementHappened || m_conditionedReflexProceed) && accumTested < accumTestedMax /*&& dendriteShifted < dendriteShiftedMax*/)
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
					//dendriteShifted += abs(newDendriteIndex - dendriteIndex);
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
					//dendriteShifted += newDendriteIndex;
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

void ConditionedReflexCreatorNeuron::Tick()
{
	if (m_reinforcementsCount < NervousSystem::Instance()->GetReinforcementCount())
	{
		++m_reinforcementsCount;
		if (m_conditionedReflexProceed)
		{
		}
		else
		{
			m_conditionedReflexCurrent->Init(m_dendrite, m_excitation);
			AccumulateExcitation(true); // accumulate all excitations
			m_prognosticCurrent->Init(m_dendrite, m_excitation);
			m_conditionedReflexCurrent->SetPrognosticNeuron(m_prognosticCurrent);
			++m_prognosticCurrent;
			++m_conditionedReflexCurrent;
			assert(m_conditionedReflexCurrent < m_conditionedReflexEnd); // TMP
			assert(m_prognosticCurrent < m_prognosticEnd); // TMP
		}
	}
	else
	{
		if (!m_conditionedReflexProceed)
		{
			m_conditionedReflexProceed.store(m_conditionedReflexProceedIn.load());
		}
		else
		{
			m_conditionedReflexProceed.load()->Tick();
		}
		AccumulateExcitation(false);
	}
	int isTimeOdd = NSNamespace::GetNSTime() % 2;
	m_dendriteOut[isTimeOdd] = m_dendrite;
	m_excitationOut[isTimeOdd] = m_excitation;
}

void ConditionedReflexCreatorNeuron::SetConditionedReflex(ConditionedReflexNeuron *reflex)
{
	m_conditionedReflexProceedIn.store(reflex);
}

void ConditionedReflexCreatorNeuron::FinishConditionedReflex(ConditionedReflexNeuron *reflex)
{
	assert(m_conditionedReflexProceed == reflex);
	if (m_conditionedReflexProceed == reflex)
	{
		m_conditionedReflexProceed = nullptr;
		m_conditionedReflexProceedIn = nullptr;
		m_reinforcementsCount = NervousSystem::Instance()->GetReinforcementCount();
	}
}

ConditionedReflexNeuron* ConditionedReflexCreatorNeuron::GetConditionedReflexProceed() const
{
	return m_conditionedReflexProceed.load();
}

void ConditionedReflexNeuron::Init(std::array<uint32_t, CONDITIONED_REFLEX_DENDRITES_NUM> &dendrite, std::array <uint16_t, CONDITIONED_REFLEX_DENDRITES_NUM> &accumulatedExcitation)
{
	m_dendrite = dendrite;
	m_excitation = accumulatedExcitation;
	m_isInitialized = true;
	m_proceedTimeMax = MOTOR_NEURON_SPONTANEOUS_ACTIVITY_TIME_DURATION;
}

void ConditionedReflexNeuron::Tick()
{
	if (m_isInitialized)
	{
		if (this == NSNamespace::GetConditionedReflexCreatorNeuron()->GetConditionedReflexProceed())
		{
			for (int ii = m_dendrite.size() - 1; ii >= 0; --ii)
			{
				ExcitationAccumulatorNeuron *neuron = (ExcitationAccumulatorNeuron*)NSNamespace::GetNeuronInterface(m_dendrite[ii]);
				if (neuron->IsMotorNeuron())
				{
					uint32_t motorNeuronId = neuron->GetDendriteNeuron();
					MotorNeuron *motorNeuron = (MotorNeuron*)NSNamespace::GetNeuronInterface(motorNeuronId);
					motorNeuron->ExcitatorySynapse();
				}
			}
			ConditionedReflexDendritesArray dendritesArray = NSNamespace::GetConditionedReflexCreatorNeuron()->GetDendritesArray();
			--m_proceedTime;
			if (!m_proceedTime)
			{
				NSNamespace::GetConditionedReflexCreatorNeuron()->FinishConditionedReflex(this);
				NervousSystem::Instance()->SetStatus(NervousSystem::NervousSystemStatus::Relaxing);
			}
		}
		else
		{ // check if conditioned reflex need to launch
			ConditionedReflexDendritesArray dendritesCur = NSNamespace::GetConditionedReflexCreatorNeuron()->GetDendritesArray();
			ConditionedReflexExitationArray exitationCur = NSNamespace::GetConditionedReflexCreatorNeuron()->GetExcitationArray();
			uint32_t error = 0;
			for (int ii = m_dendrite.size() - 1; ii >= 0; --ii)
			{
				ExcitationAccumulatorNeuron *neuron = (ExcitationAccumulatorNeuron*)NSNamespace::GetNeuronInterface(m_dendrite[ii]);
				if (neuron->IsMotorNeuron())
				{
					continue;
				}
				uint32_t errorCur = 1000 * 1000;
				for (int jj = dendritesCur.size() - 1; jj >= 0; --jj)
				{
					if (m_dendrite[ii] == dendritesCur[jj])
					{
						errorCur = (m_excitation[ii] - exitationCur[jj])*(m_excitation[ii] - exitationCur[jj]);
					}
				}
				error += errorCur;
			}
			error /= m_excitation.size() * 100;
			NervousSystem::Instance()->SetConditionedTmpStat(error);
			uint32_t prognosticReinforcement = m_prognosticNeuron->GetPrognosticReinforcement();
			if (error < 5000 && prognosticReinforcement > CONDITIONED_REFLEX_DENDRITES_NUM / 4)
			{
				NSNamespace::GetConditionedReflexCreatorNeuron()->SetConditionedReflex(this);
				m_proceedTime = m_proceedTimeMax;
				NervousSystem::Instance()->SetStatus(NervousSystem::NervousSystemStatus::ConditionedReflexProceed);
			}
		}
	}
	//assert(m_debugCheckTime != NSNamespace::GetNSTime()); // to check Tick calls once per quantum of time
	//m_debugCheckTime = NSNamespace::GetNSTime();
}

void ConditionedReflexNeuron::SetPrognosticNeuron(const PrognosticNeuron *prognosticNeuron)
{
	m_prognosticNeuron = prognosticNeuron;
}

void ConditionedReflexContainerNeuron::Init(ConditionedReflexNeuron *begin, ConditionedReflexNeuron *end)
{
	m_conditionedReflexBegin = begin;
	m_conditionedReflexCurrent = begin;
	m_conditionedReflexEnd = end;
}

void ConditionedReflexContainerNeuron::Tick()
{
	if (!NervousSystem::Instance()->IsReinforcementLevelLow() || NSNamespace::GetConditionedReflexCreatorNeuron()->GetConditionedReflexProceed())
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

uint32_t PrognosticNeuron::GetPrognosticReinforcement() const
{
	uint32_t prognosticReinforcement = 0;
	for (const auto &dendrite : m_dendrite)
	{
		assert(dynamic_cast<ExcitationAccumulatorNeuron*>(NSNamespace::GetNeuronInterface(dendrite)));
		ExcitationAccumulatorNeuron *neuron = (ExcitationAccumulatorNeuron*)NSNamespace::GetNeuronInterface(dendrite);
		prognosticReinforcement += neuron->IsReinforcementActive();
	}
	return prognosticReinforcement;
}
