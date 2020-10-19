
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
	void GetStatisticsParams(int32_t &reinforcementLevelStat, int32_t &reinforcementsCountStat) const;
	int32_t GetReinforcementCount() const; // thread safe changed on NextTick
	uint64_t GetTime() const;
	void NextTick(uint64_t timeOfTheUniverse);

	void PhotonReceived(uint8_t m_posX, uint8_t m_posY, PPh::EtherColor m_color);

private:
	int32_t GetReinforcementLevel() const;

	int32_t m_reinforcementLevel = 30000;
	int32_t m_reinforcementLevelLast = 30000;
	bool m_reinforcementZeroTouched = false;
	uint32_t m_reinforcementLevelSub = 0;

	// statistics
	int32_t m_reinforcementLevelStat = 0;
	int32_t m_reinforcementsCountStat = 0;
};

