#include <string>
#include <vector>
#include <chrono>

#include "omp.h"


float get_sim_normed(const std::string& a, const std::string& b);
float get_sim_normed_hirsch(const std::string& a, const std::string& b);
float get_sim_normed_indel(const std::string& a, const std::string& b);
float get_sim_normed_indel_cached(char(freq_table)[128], int a_len, const std::string& b);
std::vector<int> get_topk_strings(std::string a, std::vector<std::string> strings, int k);
std::vector<int> get_topk_strings_all(
		std::vector<std::string> query_strings, 
		std::vector<std::string> search_strings, 
		int k
		);
std::vector<int> get_dedup_candidates(std::vector<std::string> strings, int k);
