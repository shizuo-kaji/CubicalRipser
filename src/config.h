#ifndef CONFIG_H
#define CONFIG_H

#include <cfloat>
#include <cstdint>
#include <string>

enum calculation_method { LINKFIND, COMPUTEPAIRS, ALEXANDER};
enum output_location { LOC_NONE, LOC_YES};
enum file_format { DIPHA, PERSEUS, NUMPY, CSV };


struct Config {
	std::string filename = "";
	std::string output_filename = "output.csv"; //default output filename
	file_format format;
	calculation_method method = LINKFIND;
	double threshold = DBL_MAX;
	int maxdim=3;  // compute PH up to this dimension
	bool print = false; // flag for printing persistence pairs to stdout
	bool verbose = false;
	bool tconstruction = false; // T-construction or V-construction
	bool embedded = false; // embed image in the sphere (for alexander duality)
	output_location location = LOC_YES; // flag for saving location
	int min_recursion_to_cache = 0; // num of minimum recursions for a reduced column to be cached
	uint32_t cache_size = 1 << 31; // the maximum number of reduced columns to be cached
	int maxiter = 1000000; // maximum number of iterations for each column (for debug)
	// Experimental path for a chunked column-reduction rewrite.
	bool experimental_chunked_reduction = false;
	uint32_t experimental_chunk_size = 262144;
	uint32_t experimental_window_size = 8192;
	uint32_t experimental_relax_rounds = 0;
	// 0 means auto-detect from hardware_concurrency.
	uint32_t experimental_chunk_workers = 0;
	// 0 means "no override". 1 means force single-thread mode.
	uint32_t n_jobs = 0;
};

#endif
