#pragma once

#include "../types/real.h"

typedef struct SaveInterval
{
	real interval;
	int  count;

    SaveInterval() = default;

	SaveInterval
    (
        const char* input_filename,
        const char* interval_type
    )
    {
        real interval = C(0.0);
    
        char str[255]{'\0'};
        char buf[64]{'\0'};
        
        int num_char_interval_id = 0;
    
        while ( *(interval_type + num_char_interval_id) != '\0' ) num_char_interval_id++;
        
        FILE* fp = fopen(input_filename, "r");
    
        if (NULL == fp)
        {
            fprintf(stderr, "Error opening input file for setting save interval: %s, file: %s, line: %d.\n", interval_type, __FILE__, __LINE__);
            exit(-1);
        }
    
        while ( strncmp(buf, interval_type, num_char_interval_id) )
        {
            if ( NULL == fgets(str, sizeof(str), fp) )
            {
                fprintf(stdout, "No %s found in input file, not saving any associated data.\n", interval_type);
                fclose(fp);
    
                this->interval = C(9999.0);
                this->count    = 9999;
                return;
            }
    
            sscanf(str, "%s %" NUM_FRMT, buf, &interval);
        }
    
        this->interval = interval;
        this->count    = 0;
    }
    
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