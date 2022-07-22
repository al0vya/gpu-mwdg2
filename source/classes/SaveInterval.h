#pragma once

#include "real.h"

typedef struct SaveInterval
{
	const real interval;
	int        count;

	bool save(real current_time)
	{
		if (current_time >= interval * count)
		{
			count++;
			return true;
		}
		
		return false;
	}

} SaveInterval;