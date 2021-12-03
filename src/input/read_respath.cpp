#include "read_respath.h"

// returning C strings: https://stackoverflow.com/questions/25798977/returning-string-from-c-function
void read_respath
(
	const char* input_file,
	      char* respath
)
{
	FILE* fp = fopen(input_file, "r");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening input file for results path.\n.");
		exit(-1);
	}

	char line[255];
	char buf1[32];
	char buf2[255];

	while ( strncmp(buf1, "respath", 7) )
	{
		if ( NULL == fgets(line, sizeof(line), fp) )
		{
			fprintf(stderr, "Error reading input file for results path.\n.");
			fclose(fp);
			exit(-1);
		}

		int num_read = sscanf(line, "%s %s", buf1, buf2);
	}

	fclose(fp);

	printf("Results will be saved to this path: %s\n", buf2);
	
	int i = 0;
	
	while (true)
	{
		if (buf2[i] == '\0')
		{
			respath[i]     = '\\';
			respath[i + 1] = '\0';
			return;
		}

		respath[i] = buf2[i];

		i++;
	}
}