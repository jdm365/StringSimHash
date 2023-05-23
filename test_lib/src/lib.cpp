#include <string>
#include <vector>
#include <chrono>

#include "omp.h"

#include "lib.h"


float get_sim_normed(const std::string& a, const std::string& b) {
	int m = a.length();
	int n = b.length();

	int dp[m+1][n+1];

	for (int idx = 0; idx <= m; ++idx) {
		dp[idx][0] = idx;
	}

	for (int jdx = 0; jdx <= n; jdx++) {
		dp[0][jdx] = jdx;
	}

	for (int idx = 1; idx <= m; idx++) {
		for (int jdx = 1; jdx <= n; jdx++) {
			int sub_cost = dp[idx - 1][jdx - 1] + (a[idx - 1] != b[jdx - 1]);
			int del_ins_cost = std::min(dp[idx - 1][jdx], dp[idx][jdx - 1]) + 1;
			dp[idx][jdx] = std::min(sub_cost, del_ins_cost);
		}
	}
	// Normalize
	return 1.0f - ((float)dp[m][n] / (float)std::max(m, n));
}


float get_sim_normed_indel(const std::string& a, const std::string& b) {
	char freq_table[128] = {0};

	for (char c: a) {
		++freq_table[(int)c];
	}

	for (char c: b) {
		--freq_table[(int)c];
	}

	char distance = 0;
	for (int idx = 0; idx < 128; ++idx) {
		distance += std::abs(freq_table[idx]);
	}

	return 1.0f - ((float)distance / (float)(2 * std::max(a.length(), b.length())));
}



std::vector<int> get_topk_strings(std::string a, std::vector<std::string> strings, int k) {
	std::vector<int> topk_idxs;
	topk_idxs.reserve(k);

	float topk_distances[k];

	float distance;

	for (int idx = 0; idx < (int)strings.size(); ++idx) {
		distance = get_sim_normed(a, strings[idx]);

		if (idx < k) {
			topk_idxs[idx] = idx;
			topk_distances[idx] = distance;
			continue;
		}

		for (int jdx = 0; jdx < k; ++jdx) {
			if (distance > topk_distances[jdx]) {
				topk_idxs[jdx] = idx;
				topk_distances[jdx] = distance;
				break;
			}
		}
	}
	return topk_idxs;
}

std::vector<int> get_topk_strings_all(
		std::vector<std::string> query_strings, 
		std::vector<std::string> search_strings, 
		int k
		) {
	int n_queries  = query_strings.size();
	int n_searches = search_strings.size();

	std::vector<int> topk_idxs(n_queries * k);


	omp_set_num_threads(24);
	#pragma omp parallel for
	for (int idx = 0; idx < n_queries; ++idx) {
		float topk_distances[k];
		int   _topk_idxs[k];
		float distance;

		for (int i = 0; i < k; ++i) {
			topk_distances[i] = std::numeric_limits<float>::min();
		}

		for (int jdx = 0; jdx < n_searches; ++jdx) {
			// distance = get_sim_normed(query_strings[idx], search_strings[jdx]);
			distance = get_sim_normed_indel(query_strings[idx], search_strings[jdx]);

			if (jdx < k) {
				_topk_idxs[jdx] = jdx;
				topk_distances[jdx] = distance;
				continue;
			}

			for (int i = 0; i < k; ++i) {
				if (distance > topk_distances[i]) {
					_topk_idxs[i] = jdx;
					topk_distances[i] = distance;
					break;
				}
			}
		}

		for (int i = 0; i < k; ++i) {
			topk_idxs[idx * k + i] = _topk_idxs[i];
		}
	}
	return topk_idxs;
}



