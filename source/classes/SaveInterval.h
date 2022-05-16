#pragma once

#include "real.h"

typedef struct SaveInterval
{
	const real interval;
	int        count;

	bool save(real time_now)
	{
		if (time_now >= interval * count)
		{
			count++;
			return true;
		}
		
		return false;
	}

} SaveInterval;