#include "read_keyword_str.h"

void read_keyword_str
(
	const char* filename,
	const char* keyword,
	const int&  num_char,
	char*       value_buf
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

	char keyword_buf[128] = {'\0'};
	char line_buf   [255] = {'\0'};

	while ( strncmp(keyword_buf, keyword, num_char) )
	{
		if ( NULL == fgets(line_buf, 255, fp) )
		{
			fprintf(stderr, "Keyword %s not found when reading file %s, file: %s line: %d.\n", keyword, filename, __FILE__, __LINE__);
			value_buf[0] = '\0';
			fclose(fp);
			return;
		}

		sscanf(line_buf, "%s %s", keyword_buf, value_buf);
	}

	fclose(fp);
}