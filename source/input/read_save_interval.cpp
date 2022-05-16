#include "read_save_interval.h"

SaveInterval read_save_interval
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

			return { C(9999.0), 9999 };
		}

		sscanf(str, "%s %" NUM_FRMT, buf, &interval);
	}

	return { interval, 0 };
}