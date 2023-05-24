#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>

#include "omp.h"

#include "lib.h"



int main() {

	const int STR_LEN    = 32;
	const int n_queries  = 50000;
	const int n_searches = 50000;

	// Set seed
	srand(0);

	// Create N random strings
	std::vector<std::string> query_strings;
	for (int idx = 0; idx < n_queries; ++idx) {
		std::string s;
		for (int jdx = 0; jdx < STR_LEN; ++jdx) {
			s += (char) (rand() % 26 + 97);
		}
		query_strings.push_back(s);
	}

	std::vector<std::string> search_strings;
	for (int idx = 0; idx < n_searches; ++idx) {
		std::string s;
		for (int jdx = 0; jdx < STR_LEN; ++jdx) {
			s += (char) (rand() % 26 + 97);
		}
		search_strings.push_back(s);
	}

	// Time execution
	auto start = std::chrono::high_resolution_clock::now();

	/*
	#pragma omp parallel
	{
		#pragma omp parallel for
		for (int idx = 0; idx < N; ++idx) {
			get_topk_strings(strings[idx], strings, 50);
		}
	}
	*/
	/*
	for (int idx = 0; idx < N; ++idx) {
		get_topk_strings(strings[idx], strings, 50);
	}
	*/

	//std::vector<int> topk = get_topk_strings_all(query_strings, search_strings, 50);
	std::vector<int> topk = get_dedup_candidates(search_strings, 50);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
	std::cout << "Time: " << duration << " milliseconds" << std::endl;

	return 0;
}
