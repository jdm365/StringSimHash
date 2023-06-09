#include <string>
#include <vector>
#include <algorithm>
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

float get_sim_normed_hirsch(const std::string& a, const std::string& b) {
	if (a.length() < b.length()) {
        return get_sim_normed_hirsch(b, a);
    }

    int m = a.length();
	int n = b.length();

    //std::vector<int> prev(n + 1), curr(n + 1);
	int prev[256];
	int curr[256];

    for (int j = 0; j <= n; j++) {
        prev[j] = j;
    }

    for (int i = 1; i <= m; i++) {
        curr[0] = i;
        for (int j = 1; j <= n; j++) {
            if (a[i-1] == b[j-1]) {
                curr[j] = prev[j-1];
            } else {
                curr[j] = 1 + std::min(curr[j-1], prev[j]);
            }
        }
        std::swap(prev, curr);
    }
	return 1.0f - ((float)prev[n] / (float)m);
}


float get_sim_normed_indel(const std::string& a, const std::string& b) {
	char freq_table[128] = {0};

	for (const char c: a) {
		++freq_table[(int)c];
	}

	for (const char c: b) {
		--freq_table[(int)c];
	}

	char distance = 0;
	for (int idx = 0; idx < 128; ++idx) {
		distance += std::abs(freq_table[idx]);
	}

	return 1.0f - ((float)distance / (float)(2 * std::max(a.length(), b.length())));
}


float get_sim_normed_indel_cached(char(freq_table)[128], int a_len, const std::string& b) {
	for (const char c: b) {
		--freq_table[(int)c];
	}

	char distance = 0;
	for (int idx = 0; idx < 128; ++idx) {
		distance += std::abs(freq_table[idx]);
	}

	return 1.0f - ((float)distance / (float)(2 * std::max(a_len, (int)b.length())));
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


	#pragma omp parallel for
	for (int idx = 0; idx < n_queries; ++idx) {
		float topk_distances[k];
		int   _topk_idxs[k];
		float distance;

		for (int i = 0; i < k; ++i) {
			topk_distances[i] = std::numeric_limits<float>::min();
		}

		for (int jdx = 0; jdx < n_searches; ++jdx) {
			distance = get_sim_normed(query_strings[idx], search_strings[jdx]);
			//distance = get_sim_normed_hirsch(query_strings[idx], search_strings[jdx]);
			//distance = get_sim_normed_indel(query_strings[idx], search_strings[jdx]);

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

std::vector<int> get_dedup_candidates(std::vector<std::string> strings, int k) {
	int n_strings  = strings.size();

	std::vector<int> topk_idxs(n_strings * k);

	// Bad. Not obvious. Need to fix.
	char freq_table[128] = {0};


	#pragma omp parallel for private(freq_table)
	for (int idx = 0; idx < n_strings; ++idx) {
		float topk_distances[k];
		int   _topk_idxs[k];
		float distance;

		for (int i = 0; i < k; ++i) {
			topk_distances[i] = std::numeric_limits<float>::min();
		}

		for (int jdx = 0; jdx < n_strings; ++jdx) {
			if (idx < jdx) continue;

			// distance = get_sim_normed(query_strings[idx], search_strings[jdx]);
			if (jdx == 0) {
				char freq_table[128] = {0};

				for (char c: strings[idx]) {
					++freq_table[(int)c];
				}
			}

			//distance = get_sim_normed_indel(strings[idx], strings[jdx]);
			distance = get_sim_normed_indel_cached(freq_table, (int)strings[idx].length(), strings[jdx]);

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


