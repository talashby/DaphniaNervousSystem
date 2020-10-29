
#pragma once

#include "ParallelPhysics/PPhHelpers.h"

class Neuron;
class ConditionedReflexCreatorNeuron;

namespace NSNamespace
{
	uint32_t GetNeuronIndex(Neuron *neuron);
	uint64_t GetNSTime();
	void AddReinforcement(uint32_t val);
	Neuron* GetNeuronInterface(uint32_t neuronId);
	ConditionedReflexCreatorNeuron* GetConditionedReflexCreatorNeuron();
	uint32_t GetAccumulatorExitationNeuronNum();
}

class NervousSystem
{
public:
	static void Init();
	static NervousSystem* Instance();
	NervousSystem() = default;
	virtual ~NervousSystem() = default;

	void StartSimulation(uint64_t timeOfTheUniverse);
	void StopSimulation();
	bool IsSimulationRunning() const;
	void GetStatisticsParams(int32_t &reinforcementLevelStat, int32_t &reinforcementsCountStat, int32_t &minConditionedTmp,
		uint32_t &minNervousSystemTiming, uint32_t &maxNervousSystemTiming, uint32_t &conditionedReflexCreatorTiming) const;
	int32_t GetReinforcementCount() const; // thread safe changed on NextTick
	int32_t GetReinforcementLevel() const; // thread safe changed on NextTick
	uint64_t GetTime() const;
	void NextTick(uint64_t timeOfTheUniverse);

	void PhotonReceived(uint8_t m_posX, uint8_t m_posY, PPh::EtherColor m_color);
	bool IsReinforcementLevelLow() const;
	bool IsReinforcementHappened() const;

	void SetConditionedTmpStat(uint32_t val);

	enum class NervousSystemStatus
	{
		Relaxing = 0,
		SpontaneousActivity,
		ConditionedReflexProceed
	};

	const char* GetStatus() const;
	void SetStatus(NervousSystemStatus status);

private:
	//int32_t GetReinforcementLevel() const;

	int32_t m_reinforcementLevel = 30000;
	int32_t m_reinforcementLevelLast = 30000;
	//bool m_reinforcementHappened = false; // true after reinforcement and before zero touched
	bool m_reinforcementZeroTouched = false;
	uint32_t m_reinforcementLevelSub = 0;

	// statistics
	int32_t m_reinforcementLevelStat = 0;
	int32_t m_reinforcementsCountStat = 0;
	int32_t m_minConditionedTmpStat = -1;
	int64_t m_lastTime;
	uint64_t m_lastTimeUniverse;
	uint32_t m_quantumOfTimePerSecond;
};

