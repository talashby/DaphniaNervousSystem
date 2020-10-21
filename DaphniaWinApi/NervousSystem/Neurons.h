#pragma once

#include "ParallelPhysics/ServerProtocol.h"
#include <array>
#include <atomic>

class ConditionedReflexNeuron;

constexpr int32_t CONDITIONED_REFLEX_DENDRITES_NUM = 20; // units
constexpr uint32_t CONDITIONED_REFLEX_PER_CONTAINER = 100; // units
constexpr uint32_t SECOND_IN_QUANTS = PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND;  // quantum of time
constexpr uint32_t MILLISECOND_IN_QUANTS = PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000;  // quantum of time

class Neuron
{
public:
	Neuron() = default;
	virtual ~Neuron() = default;

	virtual void Init() {}
	virtual void Tick() {}

	virtual bool IsActive() const;

	virtual uint8_t GetType() = 0;
	uint32_t GetId();

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
		ConditionedReflexContainerNeuron,
		PrognosticNeuron
	};
};

class SensoryNeuron : public Neuron
{
public:
	SensoryNeuron() = default;
	virtual ~SensoryNeuron() = default;

	void Init()  override;

	bool IsActive() const override;
	void ExcitatorySynapse();
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;

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

	void Tick() override;
};

class SensoryNeuronGreen : public SensoryNeuron
{
public:
	SensoryNeuronGreen() = default;
	virtual ~SensoryNeuronGreen() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuronGreen); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;
};

class SensoryNeuronBlue : public SensoryNeuron
{
public:
	SensoryNeuronBlue() = default;
	virtual ~SensoryNeuronBlue() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::SensoryNeuronBlue); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Tick() override;
	bool IsActive() const override;
};

class MotorNeuron : public Neuron
{
public:
	MotorNeuron() = default;
	virtual ~MotorNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::MotorNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }
	void Init() override;

	bool IsActive() const override;

	void Tick() override;

private:
	uint8_t m_dendrite[2]; // 0-254 - excitation 255 - connection lost
	uint8_t m_axon[2]; // 0-254 - excitation 255 - connection lost
	uint16_t m_accumulatedExcitation;
	uint64_t m_lastExcitationTime;
	uint32_t m_spontaneusActivityTimeStart;
	uint64_t m_spontaneusActivityTimeFinishAbs;
	bool m_isActive[2];
};

class ExcitationAccumulatorNeuron : public Neuron
{
public:
	ExcitationAccumulatorNeuron() = default;
	virtual ~ExcitationAccumulatorNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::ExcitationAccumulatorNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }
	void Init(uint32_t dendrite);
	bool IsMotorNeuron() const;

	void Tick() override;

	uint16_t GetAccumulatedExcitation();

	void SetReflexCreatorDendriteIndex(int32_t index); // -1 not connected // used by ConditionedReflexCreatorNeuron only

	int32_t GetReflexCreatorDendriteIndex() const; // used by ConditionedReflexCreatorNeuron only
private:
	uint32_t m_dendrite; // read corresponding axon
	uint16_t m_accumulatedExcitation; // max: ExcitationAccumulationTime * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 quantum of time
	uint16_t m_accumulatedExcitationOut[2];
	uint16_t m_periodWithoutExcitation; // quantum of time
	int32_t m_reflexCreatorDendriteIndex; // used by ConditionedReflexCreatorNeuron only
};

typedef std::array<uint32_t, CONDITIONED_REFLEX_DENDRITES_NUM> ConditionedReflexDendritesArray;
typedef std::array <uint16_t, CONDITIONED_REFLEX_DENDRITES_NUM> ConditionedReflexExitationArray;

class ConditionedReflexCreatorNeuron : public Neuron
{
public:
	ConditionedReflexCreatorNeuron() = default;
	virtual ~ConditionedReflexCreatorNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::ConditionedReflexCreatorNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }
	void Init(ExcitationAccumulatorNeuron *begin, ExcitationAccumulatorNeuron *end,
		ConditionedReflexNeuron *beginCond, ConditionedReflexNeuron *endCond);
	ConditionedReflexDendritesArray GetDendritesArray() const; // for other threads
	ConditionedReflexExitationArray GetExcitationArray() const; // for other threads
	void Tick() override;

private:
	ConditionedReflexDendritesArray m_dendrite; // read corresponding axon
	ConditionedReflexDendritesArray m_dendriteOut[2];
	ConditionedReflexExitationArray m_excitation; // max: ExcitationAccumulationTime * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 quantum of time
	ConditionedReflexExitationArray m_excitationOut[2];
	ExcitationAccumulatorNeuron *m_accumulatorBegin;
	ExcitationAccumulatorNeuron *m_accumulatorEnd;
	ExcitationAccumulatorNeuron *m_accumulatorCurrent;
	ConditionedReflexNeuron *m_conditionedReflexBegin;
	ConditionedReflexNeuron *m_conditionedReflexCurrent;
	ConditionedReflexNeuron *m_conditionedReflexEnd;
	int32_t m_reinforcementsCount;
};

class ConditionedReflexNeuron : public Neuron
{
public:
	ConditionedReflexNeuron() = default;
	virtual ~ConditionedReflexNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::ConditionedReflexNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Init(std::array<uint32_t, CONDITIONED_REFLEX_DENDRITES_NUM> &dendrite, std::array <uint16_t, CONDITIONED_REFLEX_DENDRITES_NUM> &accumulatedExcitation);

	void Tick() override;
private:
	std::atomic<bool> m_isActive;
	std::array<uint32_t, CONDITIONED_REFLEX_DENDRITES_NUM> m_dendrite; // read corresponding axon
	std::array <uint16_t, CONDITIONED_REFLEX_DENDRITES_NUM> m_excitation; // max: ExcitationAccumulationTime * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 quantum of time
	//uint64_t m_debugCheckTime = -1; // to check Tick calls once per quantum of time
};

class ConditionedReflexContainerNeuron : public Neuron
{
public:
	ConditionedReflexContainerNeuron() = default;
	virtual ~ConditionedReflexContainerNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::ConditionedReflexContainerNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Init(ConditionedReflexNeuron *begin, ConditionedReflexNeuron *end);
	void Tick() override;
private:
	ConditionedReflexNeuron *m_conditionedReflexBegin;
	ConditionedReflexNeuron *m_conditionedReflexEnd;
	ConditionedReflexNeuron *m_conditionedReflexCurrent;
};

class PrognosticNeuron : public Neuron
{
public:
	PrognosticNeuron() = default;
	virtual ~PrognosticNeuron() = default;
	constexpr static uint8_t GetTypeStatic() { return static_cast<uint8_t>(NeuronTypes::PrognosticNeuron); }
	uint8_t GetType() override { return GetTypeStatic(); }

	void Init(std::array<uint32_t, CONDITIONED_REFLEX_DENDRITES_NUM> &dendrite, std::array <uint16_t, CONDITIONED_REFLEX_DENDRITES_NUM> &accumulatedExcitation);

	//void Tick() override;
private:
	std::atomic<bool> m_isActive;
	std::array<uint32_t, CONDITIONED_REFLEX_DENDRITES_NUM> m_dendrite; // read corresponding axon
	std::array <uint16_t, CONDITIONED_REFLEX_DENDRITES_NUM> m_excitation; // max: ExcitationAccumulationTime * PPh::CommonParams::QUANTUM_OF_TIME_PER_SECOND / 1000 quantum of time
};

