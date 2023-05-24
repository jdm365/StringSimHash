#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "omp.h"

#include "../include/lib.h"


float get_sim_normed_wrapper(const std::string& a, const std::string& b) {
    return get_sim_normed(a, b);
}

float get_sim_normed_indel_wrapper(const std::string& a, const std::string& b) {
    return get_sim_normed_indel(a, b);
}

std::vector<int> get_topk_strings_wrapper(std::string a, std::vector<std::string> strings, int k) {
    return get_topk_strings(a, strings, k);
}

std::vector<int> get_topk_strings_all_wrapper(
        std::vector<std::string> query_strings,
        std::vector<std::string> search_strings,
        int k
    ) {
    return get_topk_strings_all(query_strings, search_strings, k);
}

std::vector<int> get_dedup_candidates_wrapper(std::vector<std::string> strings, int k) {
	return get_dedup_candidates(strings, k);
}


PYBIND11_MODULE(StringDedup, m) {
	omp_set_num_threads(omp_get_max_threads());

	m.doc() = "String Dedup Backend";

	m.def("get_sim_normed", &get_sim_normed_wrapper, "Get levenshtein similarity between two strings",
			pybind11::arg("a"), pybind11::arg("b"));
	m.def("get_sim_normed_indel", &get_sim_normed_indel_wrapper, "Get indel similarity between two strings",
			pybind11::arg("a"), pybind11::arg("b"));
	m.def("get_topk_strings", &get_topk_strings_wrapper, "Get topk strings",
			pybind11::arg("a"), pybind11::arg("strings"), pybind11::arg("k"));
	m.def("get_topk_strings_all", &get_topk_strings_all_wrapper, "Get the top-k most similar strings for all queries",
			pybind11::arg("query_strings"), pybind11::arg("search_strings"), pybind11::arg("k"));
	m.def("get_dedup_candidates", &get_dedup_candidates_wrapper, "Get the top-k most similar strings for all queries",
			pybind11::arg("strings"), pybind11::arg("k"));
}
