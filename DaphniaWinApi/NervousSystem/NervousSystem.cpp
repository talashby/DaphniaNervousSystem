#include "NervousSystem.h"

NervousSystem* s_nervousSystem = nullptr;

void NervousSystem::Init()
{
	if (s_nervousSystem)
	{
		delete s_nervousSystem;
	}
	else
	{
		s_nervousSystem = new NervousSystem();
	}
}

NervousSystem* NervousSystem::Instance()
{
	return s_nervousSystem;
}

void NervousSystem::PhotonReceived(uint8_t m_posX, uint8_t m_posY, PPh::EtherColor m_color)
{
}
