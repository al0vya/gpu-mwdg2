#include "read_keyword_real.h"

real read_keyword_real
(
	const char* filename,
	const char* keyword,
	const int&  num_char
)
{
	if (num_char > 128)
	{
		fprintf(stderr, "Keyword length %s exceeds keyword buffer size 255, file: %s line: %d.\n", keyword, __FILE__, __LINE__);
		exit(-1);
	}
	
	FILE* fp = fopen(filename, "r");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening file %s for reading keyword %s, file: %s line: %d.\n", filename, keyword, __FILE__, __LINE__);
		exit(-1);
	}

	real value = C(5.0);

	char keyword_buf[128] = {'\0'};
	char line_buf   [255] = {'\0'};

	while ( strncmp(keyword_buf, keyword, num_char) )
	{
		if ( NULL == fgets(line_buf, 255, fp) )
		{
			fprintf(stderr, "Keyword %s not found when reading file %s, file: %s line: %d.\n", keyword, filename, __FILE__, __LINE__);
			fclose(fp);
			return C(0.0);
		}

		sscanf(line_buf, "%s %" NUM_FRMT, keyword_buf, &value);
	}

	fclose(fp);

	return value;
}