
#pragma once

#include "ParallelPhysics/PPhHelpers.h"

class Neuron
{
	Neuron() = default;
	virtual ~Neuron() = default;
};

class SimpleAdder : public Neuron
{
};

class NervousSystem
{
public:
	static void Init();
	static NervousSystem* Instance();
	NervousSystem() = default;
	virtual ~NervousSystem() = default;

	void PhotonReceived(uint8_t m_posX, uint8_t m_posY, PPh::EtherColor m_color);

private:
	
};

