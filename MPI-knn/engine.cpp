#include <vector>
#include <queue>
#include <mpi.h>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include "engine.h"
#include "common.h"
#include <cmath>
#include <memory>
#include <iostream>
#include <string>
#include <unistd.h> // for gethostname

// --- USER DEFINED GLOBAL VARIABLES ---
int QBLOCK = 64; 

using namespace std;

using Pair = pair<double, int>; // {distance_squared, global_data_index}

// Comparator for top-k max-heap with worst element on top
struct WorstOnTop {
    bool operator()(const Pair& a, const Pair& b) const {
        if (a.first != b.first) return a.first < b.first;
        return a.second > b.second;
    }
};

using MaxPQ = priority_queue<Pair, vector<Pair>, WorstOnTop>;

static inline bool better_than(const Pair& a, const Pair& b) {
    if (a.first != b.first) return a.first < b.first;
    return a.second > b.second;
}

// NEW: Overloaded Euclidean distance for pointer-to-vector (used in fast tree searching)
static inline double eu_dist_sq_ptr(const double* a, const vector<double>& b, int A) {
    double sum = 0.0;
    for (int i = 0; i < A; ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// --- KD-TREE ---
struct KDTreeNode {
    DataPoint point;  
    int id;           
    int axis;         
    unique_ptr<KDTreeNode> left;
    unique_ptr<KDTreeNode> right;

    KDTreeNode(const DataPoint& p, int ax, int global_id) 
        : point(p), id(global_id), axis(ax), left(nullptr), right(nullptr) {}
};

static unique_ptr<KDTreeNode> build_kdtree(vector<DataPoint>& points, int start, int end, int depth) {
    if (start >= end) return nullptr; 

    int dimensions = points[0].attrs.size();
    int axis = depth % dimensions; 
    int mid = start + (end - start) / 2;

    nth_element(points.begin() + start, 
                points.begin() + mid, 
                points.begin() + end,
                [axis](const DataPoint& a, const DataPoint& b) {
                    return a.attrs[axis] < b.attrs[axis];
                });

    auto node = make_unique<KDTreeNode>(points[mid], axis, points[mid].id);
    node->left = build_kdtree(points, start, mid, depth + 1);
    node->right = build_kdtree(points, mid + 1, end, depth + 1);
    return node;
}

// --- WRAPPERS ---
static unique_ptr<KDTreeNode> construct_kdtree(vector<DataPoint> points) {
    if (points.empty()) return nullptr;
    return build_kdtree(points, 0, points.size(), 0);
}

// --- SEARCH FUNCTIONS ---
// MODIFIED: Accept raw pointer t_attrs and integer k
static void search_kdtree(int k, const double* t_attrs, int A, MaxPQ& pq, const KDTreeNode* node) {
    if (node == nullptr) return;

    double dist_to_node = eu_dist_sq_ptr(t_attrs, node->point.attrs, A);

    if (pq.size() < k || dist_to_node < pq.top().first) {
        pq.push({dist_to_node, node->id});
        if (pq.size() > k) pq.pop(); 
    }

    int axis = node->axis;
    double diff = t_attrs[axis] - node->point.attrs[axis];

    const KDTreeNode* first_child = diff < 0 ? node->left.get() : node->right.get();
    const KDTreeNode* second_child = diff < 0 ? node->right.get() : node->left.get();

    search_kdtree(k, t_attrs, A, pq, first_child);

    double plane_dist_sq = diff * diff;
    if (pq.size() < k || plane_dist_sq < pq.top().first) {
        search_kdtree(k, t_attrs, A, pq, second_child);
    }
}

// merge arrays and return top-k
static vector<Pair> merge_topk_sorted(const vector<Pair>& a, const vector<Pair>& b, int k) {
    vector<Pair> out;
    out.reserve(k);
    size_t i = 0, j = 0;
    while ((int)out.size() < k && (i < a.size() || j < b.size())) {
        if (j == b.size() || (i < a.size() && better_than(a[i], b[j]))) {
            out.push_back(a[i++]);
        } else {
            out.push_back(b[j++]);
        }
    }
    return out;
}

// MODIFIED: Combined Top-K Function
static void compute_topk(int routine, const KDTreeNode* kd_root,
                         int A, int local_N, int local_start,
                         int block_Q, int q_start, 
                         const vector<double>& local_flat_d,
                         const vector<double>& local_flat_q,
                         const vector<int>& local_query_ks, 
                         vector<vector<Pair>>& block_topk) {
    vector<MaxPQ> heaps(block_Q);

    if (routine == 0) {
        // Brute Force: Data-Outer Loop for Cache Optimization
        for (int li = 0; li < local_N; li++) {
            const double* dp = &local_flat_d[(long long)li * A];
            const int gi = local_start + li;

            for (int ql = 0; ql < block_Q; ql++) {
                const int kq = local_query_ks[q_start + ql]; 
                if (kq == 0) continue;

                MaxPQ& h = heaps[ql];
                const double* qr = &local_flat_q[(long long)(q_start + ql) * A];

                const double threshold = ((int)h.size() == kq) ? h.top().first : numeric_limits<double>::max();

                double dist = 0.0;
                bool pruned = false;

                for (int a = 0; a < A; a++) {
                    double d = dp[a] - qr[a];
                    dist += d * d;
                    if (dist > threshold) {
                        pruned = true;
                        break;
                    }
                }

                if (pruned) continue;

                Pair cand = {dist, gi};
                
                if ((int)h.size() < kq) {
                    h.push(cand);
                } else if (better_than(cand, h.top())) {
                    h.pop();
                    h.push(cand);
                }
            }
        }
    } else {
        // Tree Search: Query-Outer Loop for Sequential Traversal
        for (int ql = 0; ql < block_Q; ql++) {
            const int kq = local_query_ks[q_start + ql]; 
            if (kq == 0) continue;

            MaxPQ& h = heaps[ql];
            const double* qr = &local_flat_q[(long long)(q_start + ql) * A];

            search_kdtree(kq, qr, A, h, kd_root);
        }
    }

    // Convert local heaps to sorted vectors in final order
    for (int ql = 0; ql < block_Q; ql++) {
        auto& h = heaps[ql];
        auto& v = block_topk[ql];
        v.reserve(h.size());
        while (!h.empty()) {
            v.push_back(h.top());
            h.pop();
        }
        sort(v.begin(), v.end(), better_than);
    }
}

// Helper to find the majority class label
static int get_best_label(const vector<Pair>& result, const vector<DataPoint>& dataset) {
    unordered_map<int, int> freq;
    for (const auto& pr : result) {
        freq[dataset[pr.second].label]++;
    }

    int best_label = -1;
    int best_count = -1;

    for (const auto& it : freq) {
        int lbl = it.first;
        int cnt = it.second;
        if (cnt > best_count || (cnt == best_count && lbl > best_label)) {
            best_count = cnt;
            best_label = lbl;
        }
    }
    return best_label;
}


void Engine::KNN(Params& p, vector<DataPoint>& dataset, vector<Query>& queries) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef DEBUG
    // --- TIMING VARIABLES ---
    double t_pcomp = 0.0, t_data = 0.0, t_query = 0.0, t_build = 0.0, t_compute = 0.0, t_reduce = 0.0, t_sendback = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_pcomp = MPI_Wtime();
#endif

    MPI_Bcast(&p.num_data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p.num_queries, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p.num_attrs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    const int N = p.num_data;
    const int Q = p.num_queries;
    const int A = p.num_attrs;

    // Hardcode the mapping from hostname to node ID
    unordered_map<string, int> host_to_node = {
        {"soctf-pdc-008", 0},  {"soctf-pdc-012", 1},  {"soctf-pdc-015", 2},
        {"soctf-pdc-016", 3},  {"soctf-pdc-018", 4},  {"soctf-pdc-019", 5},
        {"soctf-pdc-020", 6},  {"soctf-pdc-027", 7},  {"soctf-pdc-029", 8},
        {"soctf-pdc-030", 9},  {"soctf-pdc-031", 10}, {"soctf-pdc-032", 11},
        {"soctf-pdc-033", 12}, {"soctf-pdc-034", 13}, {"soctf-pdc-035", 14},
        {"soctf-pdc-036", 15}, {"soctf-pdc-037", 16}, {"soctf-pdc-038", 17},
        {"soctf-pdc-039", 18}, {"soctf-pdc-040", 19}
    };

    // 1. Use the MPI standard buffer size to guarantee no buffer overflows
    char name_buffer[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(name_buffer, &name_len);

    // 2. Immediately convert to a modern C++ string
    string hostname(name_buffer, name_len);

    // 3. Normalize the hostname (Safest approach for HPC clusters)
    // This strips ".comp.nus.edu.sg" if it exists, leaving just "soctf-pdc-036"
    // This ensures your map keys only ever need to be the short, simple names!
    size_t dot_pos = hostname.find('.');
    if (dot_pos != string::npos) {
        hostname = hostname.substr(0, dot_pos);
    }

    int node_id = host_to_node.at(hostname);
    
    // Master rank will gather all node IDs
    vector<int> node_ids;
    if (rank == 0) {
        node_ids.resize(size); 
    }

    // All ranks send their node_id to master (rank 0)
    MPI_Gather(&node_id, 1, MPI_INT, 
               node_ids.data(), 1, MPI_INT, 
               0, MPI_COMM_WORLD);

    int config_data[3];

    if (rank == 0) {
        // Initialize 2D vector for latencies (Hardcoded from CSV data to 4dp)
        const vector<float> node_pair_bandwidth = {
            0.0621f, 0.6474f, 0.1680f, 1.4071f, 0.5579f, 2.2007f, 0.9433f, 0.2966f, 0.3444f, 0.8116f, 0.5579f, 0.2928f, 0.3102f, 0.4223f, 1.4071f, 1.1859f, 0.8683f, 0.7935f, 1.2211f, 0.2362f,
            0.6474f, 0.0211f, 0.1577f, 0.1627f, 0.2402f, 2.0722f, 0.2966f, 0.4234f, 0.4701f, 0.5242f, 0.5681f, 0.3501f, 0.0233f, 0.5321f, 4.6413f, 1.3192f, 0.0260f, 0.0233f, 0.0312f, 0.0411f,
            0.1680f, 0.1577f, 0.0205f, 0.1914f, 0.1313f, 1.8388f, 0.4339f, 0.4395f, 0.4774f, 0.4547f, 1.2278f, 0.4971f, 0.3611f, 0.8776f, 3.5439f, 2.2948f, 0.0399f, 0.0364f, 0.0378f, 0.0463f,
            1.4071f, 0.1627f, 0.1914f, 0.0189f, 0.2939f, 2.1405f, 0.6939f, 1.0675f, 0.2933f, 0.4553f, 0.4654f, 0.4403f, 0.1561f, 0.1843f, 5.1118f, 1.3734f, 0.0419f, 0.1028f, 0.0878f, 0.1299f,
            0.5579f, 0.2402f, 0.1313f, 0.2939f, 0.0552f, 1.2662f, 0.2966f, 0.3913f, 0.6052f, 0.4622f, 0.6849f, 0.9703f, 0.2416f, 0.2186f, 0.9252f, 0.9703f, 0.1430f, 0.4789f, 0.6238f, 0.7225f,
            2.2007f, 2.0722f, 1.8388f, 2.1405f, 1.2662f, 0.0868f, 1.5791f, 1.6579f, 1.5186f, 1.0206f, 2.1294f, 1.1551f, 1.4265f, 2.6118f, 0.2757f, 0.3820f, 2.0041f, 1.4363f, 1.4653f, 1.5194f,
            0.9433f, 0.2966f, 0.4339f, 0.6939f, 0.2966f, 1.5791f, 0.0181f, 0.3844f, 0.3047f, 0.1862f, 0.1812f, 0.1876f, 0.3348f, 0.2404f, 0.9400f, 2.8235f, 0.3834f, 0.4425f, 0.3709f, 0.2622f,
            0.2966f, 0.4234f, 0.4395f, 1.0675f, 0.3913f, 1.6579f, 0.3844f, 0.0859f, 0.0724f, 0.1002f, 0.1121f, 0.1022f, 0.2340f, 0.0000f, 3.0749f, 3.3210f, 0.1784f, 0.1619f, 0.1785f, 0.1865f,
            0.3444f, 0.4701f, 0.4774f, 0.2933f, 0.6052f, 1.5186f, 0.3047f, 0.0724f, 0.0518f, 0.0817f, 0.0853f, 0.0620f, 0.1229f, 0.0545f, 2.0933f, 3.1717f, 0.1003f, 0.1883f, 0.2109f, 0.2244f,
            0.8116f, 0.5242f, 0.4547f, 0.4553f, 0.4622f, 1.0206f, 0.1862f, 0.1002f, 0.0817f, 0.0565f, 0.1357f, 0.0735f, 0.1871f, 0.1546f, 3.0422f, 3.1247f, 0.2123f, 0.2034f, 0.1964f, 0.3909f,
            0.5579f, 0.5681f, 1.2278f, 0.4654f, 0.6849f, 2.1294f, 0.1812f, 0.1121f, 0.0853f, 0.1357f, 0.0683f, 0.1160f, 1.0442f, 0.1966f, 3.2564f, 3.1325f, 0.1859f, 0.1204f, 0.1587f, 0.2127f,
            0.2928f, 0.3501f, 0.4971f, 0.4403f, 0.9703f, 1.1551f, 0.1876f, 0.1022f, 0.0620f, 0.0735f, 0.1160f, 0.0402f, 1.4926f, 0.1324f, 3.2991f, 3.3183f, 0.2012f, 0.1639f, 0.1322f, 0.2262f,
            0.3102f, 0.0233f, 0.3611f, 0.1561f, 0.2416f, 1.4265f, 0.3348f, 0.2340f, 0.1229f, 0.1871f, 1.0442f, 1.4926f, 0.0312f, 0.3754f, 4.9017f, 5.0682f, 0.4069f, 0.4440f, 0.4099f, 0.3769f,
            0.4223f, 0.5321f, 0.8776f, 0.1843f, 0.2186f, 2.6118f, 0.2404f, 0.0000f, 0.0545f, 0.1546f, 0.1966f, 0.1324f, 0.3754f, 0.0245f, 4.9918f, 4.9769f, 0.4217f, 0.3603f, 0.3480f, 0.3827f,
            1.4071f, 4.6413f, 3.5439f, 5.1118f, 0.9252f, 0.2757f, 0.9400f, 3.0749f, 2.0933f, 3.0422f, 3.2564f, 3.2991f, 4.9017f, 4.9918f, 0.0297f, 0.3601f, 5.2892f, 5.2301f, 5.4428f, 5.2144f,
            1.1859f, 1.3192f, 2.2948f, 1.3734f, 0.9703f, 0.3820f, 2.8235f, 3.3210f, 3.1717f, 3.1247f, 3.1325f, 3.3183f, 5.0682f, 4.9769f, 0.3601f, 0.0276f, 5.0971f, 4.9839f, 5.0721f, 5.0181f,
            0.8683f, 0.0260f, 0.0399f, 0.0419f, 0.1430f, 2.0041f, 0.3834f, 0.1784f, 0.1003f, 0.2123f, 0.1859f, 0.2012f, 0.4069f, 0.4217f, 5.2892f, 5.0971f, 0.0311f, 0.3935f, 0.4022f, 0.3989f,
            0.7935f, 0.0233f, 0.0364f, 0.1028f, 0.4789f, 1.4363f, 0.4425f, 0.1619f, 0.1883f, 0.2034f, 0.1204f, 0.1639f, 0.4440f, 0.3603f, 5.2301f, 4.9839f, 0.3935f, 0.0320f, 0.3247f, 0.3857f,
            1.2211f, 0.0312f, 0.0378f, 0.0878f, 0.6238f, 1.4653f, 0.3709f, 0.1785f, 0.2109f, 0.1964f, 0.1587f, 0.1322f, 0.4099f, 0.3480f, 5.4428f, 5.0721f, 0.4022f, 0.3247f, 0.0281f, 0.3825f,
            0.2362f, 0.0411f, 0.0463f, 0.1299f, 0.7225f, 1.5194f, 0.2622f, 0.1865f, 0.2244f, 0.3909f, 0.2127f, 0.2262f, 0.3769f, 0.3827f, 5.2144f, 5.0181f, 0.3989f, 0.3857f, 0.3825f, 0.0301f
        };

        const float FACTOR_COMM = 0.2044f; 
        const float FACTOR_SB = 1.374f;
        const float FACTOR_RED_COMM = 0.1389f;
        const float FACTOR_RED_COMP = 0.5702f;

        // ── 1. One-Time Hardware Mapping ──────────────────────────────────────────
        int max_remote_cores = 1;
        bool is_bonus = true;
        unordered_map<int, int> core_counts;
        for (int id : node_ids) {
            core_counts[id]++;
            is_bonus &= (id == 10) || (id == 11); 
        }
        
        int node_0 = node_ids[0];
        for (const auto& pair : core_counts) {
            if (pair.first != node_0) {
                max_remote_cores = max(max_remote_cores, pair.second);
            }
        }

        struct Config {
            int d_split;
            bool flip;
            int mode;        // 0: Brute Force, 1: KD Tree
            float t_total;
        };

        int alg_mode = (int)(A < 11);

        bool split_queries = (Q * 3) >= (N * 2);
        int d_split = split_queries ? 1 : 24; // >=40% queries, split by queries, else data
        Config best_config = {d_split, !split_queries, alg_mode, numeric_limits<float>::infinity()};

        // Iteration search space
        if (is_bonus) {
            float K;
            long long sum_k = 0;
            for (int i = 0; i < 67; i++) {
                sum_k += queries[(6767 * i) % queries.size()].k;
            }
            K = (float)sum_k / 67.0f;
        
            vector<int> factors;
            vector<int> back_factors;
            for (int i = 1; i <= sqrt(size); ++i) {
                if (size % i == 0) {
                    factors.push_back(i);
                    if (i != size / i) {
                        back_factors.push_back(size / i);
                    }
                }
            }

            for (int i = back_factors.size() - 1; i >= 0; --i) {
                factors.push_back(back_factors[i]);
            }         
            
            auto get_bw = [&](int id1, int id2) {
                return node_pair_bandwidth[id1 * 20 + id2];
            };

            for (int d_split : factors) {
                for (bool flip_state : {true, false}) {
                    // S_q (Query Split) is exactly size / split
                    int q_split = size / d_split; 

                    // --- 1. Evaluate Layout & Communication ONCE per grid configuration ---
                    int rows = flip_state ? d_split : q_split;
                    int cols = flip_state ? q_split : d_split;

                    float local_N = N / (float)d_split;
                    float local_Q = Q / (float)q_split;

                    int penalty = 1;
                    if ((d_split > 1) && (d_split < size)) {
                        penalty = max_remote_cores;
                    }

                    auto simulate_comm = [&](float comm_size, bool bcast_along_rows) -> float {
                        // Simulate scatter along first row, broadcast along columns
                        // If bcast_along_rows, transpose simulation
                        int r_count = bcast_along_rows ? cols : rows;
                        int c_count = bcast_along_rows ? rows : cols;

                        auto get_rank = [&](int r, int c) {
                            return bcast_along_rows ? (c * cols + r) : (r * cols + c);
                        };

                        // Phase 1: Scatter along first row
                        float ut_scatter = 0.0f;
                        int node_00 = node_ids[get_rank(0, 0)];
                        for (int c = 1; c < c_count; ++c) {
                            ut_scatter += get_bw(node_00, node_ids[get_rank(0, c)]);
                        }

                        // Phase 2: Broadcast along columns
                        float pen = bcast_along_rows ? 1.0f : (float)penalty; // penalty along non contiguous axis
                        float ut_bcast = 0.0f;
                        for (int c = 0; c < c_count; ++c) {
                            int node_0c = node_ids[get_rank(0, c)];
                            float cost = 0.0f;
                            vector<bool> new_pair(400, true);

                            for (int r = 1; r < r_count; ++r) {
                                int node_rc = node_ids[get_rank(r, c)];

                                if (new_pair[node_0c * 20 + node_rc]) {
                                    cost += get_bw(node_0c, node_rc) * pen;
                                    new_pair[node_0c * 20 + node_rc] = false;
                                }
                            }
                            ut_bcast = max(ut_bcast, cost); 
                        }
                        return (ut_scatter + ut_bcast) * (A + 0.5f) * comm_size * FACTOR_COMM / 1e6f;
                    };
                    
                    // Calculate Initial Comm Times
                    float t_dp = simulate_comm(local_N, flip_state);
                    float t_q  = simulate_comm(local_Q, !flip_state);

                    auto get_rank = [&](int q, int d) {
                        return flip_state ? (d * q_split + q) : (q * d_split + d);
                    };

                    // Sendback
                    float ut_sb = 0.0f;
                    int node_00 = node_ids[get_rank(0, 0)];
                    for (int q = 1; q < q_split; ++q) {
                        ut_sb += get_bw(node_00, node_ids[get_rank(q, 0)]);
                    }

                    float t_sb = ut_sb * K * local_Q * FACTOR_SB / 1e5f;

                    // Reduce
                    float pen = flip_state ? (float)penalty : 1.0; // penalty along non contiguous axis
                    float ut_red_comm = 0.0f;
                    for (int q = 0; q < q_split; ++q) {
                        float cost = 0.0f;
                        for (int step = 1; step < d_split; step *= 2) {
                            for (int d = 0; d + step < d_split; d += step * 2) {
                                int node_dest = node_ids[get_rank(q, d)];
                                int node_src = node_ids[get_rank(q, d + step)];
                                cost += get_bw(node_dest, node_src) * pen;
                            }
                            
                        }
                        ut_red_comm = max(ut_red_comm, cost);
                    }
                    
                    float t_red = (ut_red_comm * FACTOR_RED_COMM + log2f(d_split) * FACTOR_RED_COMP) * K * local_Q / 1e7f;

                    // Sum all network phases
                    float t_total = t_dp + t_q + t_sb + t_red;

#ifdef DEBUG
                    // print config and predictions here
                    fprintf(stderr, "Config [d_split: %d, flip: %s, mode: %d] -> t_dp: %.4f, t_q: %.4f, t_red: %.4f, t_sb: %.4f, t_total: %.4f\n",
                            d_split, flip_state ? "true" : "false", alg_mode, t_dp, t_q, t_red, t_sb, t_total);
#endif
                
                    if (t_total < best_config.t_total) {
                        best_config = {d_split, flip_state, alg_mode, t_total};
                    }
                }
            }
        }

        config_data[0] = best_config.d_split;
        config_data[1] = best_config.flip ? 1 : 0;
        config_data[2] = best_config.mode;
    }

    // Rank 0 broadcasts the 3 integers to everyone
    MPI_Bcast(config_data, 3, MPI_INT, 0, MPI_COMM_WORLD);

    // All ranks unpack the broadcasted data
    int data_split = config_data[0];
    int data_bcast_row = config_data[1];
    int routine = config_data[2];

    // ==========================================
    // --- GRID SETUP ---
    // ==========================================
    const int num_data_splits = data_split;
    const int num_query_splits = size / data_split;
    int my_data_split, my_query_split;
    MPI_Comm shared_data_comm, shared_query_comm;

    if (data_bcast_row) {

#ifdef DEBUG
        if (rank == 0) {
            fprintf(stderr, "Grid Topology configured as: %d Data Rows x %d Query Columns\n", 
                num_data_splits, num_query_splits);
        }
#endif

        my_data_split = rank / num_query_splits;  
        my_query_split = rank % num_query_splits; 
    } else {

#ifdef DEBUG
        if (rank == 0) {
            fprintf(stderr, "Grid Topology configured as: %d Query Rows x %d Data Columns\n", 
                num_query_splits, num_data_splits);
        }
#endif

        my_data_split = rank % num_data_splits;  
        my_query_split = rank / num_data_splits; 
    }

    MPI_Comm_split(MPI_COMM_WORLD, my_data_split, my_query_split, &shared_data_comm);
    MPI_Comm_split(MPI_COMM_WORLD, my_query_split, my_data_split, &shared_query_comm);

    int shared_query_rank, shared_query_size;
    MPI_Comm_rank(shared_query_comm, &shared_query_rank); 
    MPI_Comm_size(shared_query_comm, &shared_query_size); 

    // ==========================================
    // --- 1) PARTITION DATA & QUERIES ---
    // ==========================================
    vector<int> d_counts(num_data_splits), d_displs(num_data_splits);
    vector<int> d_attr_counts(num_data_splits), d_attr_displs(num_data_splits);
    for (int r = 0; r < num_data_splits; r++) {
        int s = (long long)r * N / num_data_splits;
        int e = (long long)(r + 1) * N / num_data_splits;
        d_counts[r] = e - s;
        d_displs[r] = s;
        d_attr_counts[r] = d_counts[r] * A;
        d_attr_displs[r] = d_displs[r] * A;
    }

    vector<double> flat_d(rank == 0 ? (long long)N * A : 0);
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int a = 0; a < A; a++) flat_d[(long long)i * A + a] = dataset[i].attrs[a];
        }
    }

    vector<int> q_counts(num_query_splits), q_displs(num_query_splits);
    vector<int> q_attr_counts(num_query_splits), q_attr_displs(num_query_splits);
    for (int r = 0; r < num_query_splits; r++) {
        int s = (long long)r * Q / num_query_splits;
        int e = (long long)(r + 1) * Q / num_query_splits;
        q_counts[r] = e - s;
        q_displs[r] = s;
        q_attr_counts[r] = q_counts[r] * A;
        q_attr_displs[r] = q_displs[r] * A;
    }

    vector<int> query_ks(rank == 0 ? Q : 0);
    vector<double> flat_q(rank == 0 ? (long long)Q * A : 0);
    if (rank == 0) {
        for (int q = 0; q < Q; q++) {
            query_ks[q] = queries[q].k;
            for (int a = 0; a < A; a++) flat_q[(long long)q * A + a] = queries[q].attrs[a];
        }
    }

    const int local_N = d_counts[my_data_split];
    const int local_start = d_displs[my_data_split];
    const int local_Q = q_counts[my_query_split];

    vector<double> local_flat_d(local_N * A);
    vector<int> local_query_ks(local_Q);
    vector<double> local_flat_q(local_Q * A);
    
#ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        t_pcomp = MPI_Wtime() - t_pcomp;
        t_data = MPI_Wtime();
    }
#endif

    // ==========================================
    // --- 2) SEND DATA & QUERIES ---
    // ==========================================

    // Scatter down Columns, Broadcast across Rows
    if (my_query_split == 0) {
        MPI_Scatterv(rank == 0 ? flat_d.data() : nullptr,
                     d_attr_counts.data(), d_attr_displs.data(), MPI_DOUBLE,
                     local_flat_d.data(), local_N * A, MPI_DOUBLE, 0, shared_query_comm);
    }

    MPI_Bcast(local_flat_d.data(), local_N * A, MPI_DOUBLE, 0, shared_data_comm);

#ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        t_data = MPI_Wtime() - t_data;
        t_query = MPI_Wtime();
    }
#endif

    // Scatter across Rows, Broadcast down Columns
    if (my_data_split == 0) {
        MPI_Scatterv(rank == 0 ? query_ks.data() : nullptr,
                     q_counts.data(), q_displs.data(), MPI_INT,
                     local_query_ks.data(), local_Q, MPI_INT, 0, shared_data_comm);

        MPI_Scatterv(rank == 0 ? flat_q.data() : nullptr,
                     q_attr_counts.data(), q_attr_displs.data(), MPI_DOUBLE,
                     local_flat_q.data(), local_Q * A, MPI_DOUBLE, 0, shared_data_comm);
    }

    MPI_Bcast(local_query_ks.data(), local_Q, MPI_INT, 0, shared_query_comm);
    MPI_Bcast(local_flat_q.data(), local_Q * A, MPI_DOUBLE, 0, shared_query_comm);  

#ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        t_query = MPI_Wtime() - t_query;
        t_build = MPI_Wtime();
    }
#endif

    // ==========================================
    // --- 3) TREE INITIALIZATION ---
    // ==========================================
    unique_ptr<KDTreeNode> kd_root = nullptr;

    if (routine) {
        vector<DataPoint> local_dp(local_N);
        for (int i = 0; i < local_N; ++i) {
            local_dp[i].id = local_start + i;
            local_dp[i].label = -1;
            local_dp[i].attrs.assign(&local_flat_d[(long long)i * A], &local_flat_d[(long long)(i + 1) * A]);
        }

        kd_root = construct_kdtree(local_dp);
    }

#ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        t_build = MPI_Wtime() - t_build;
        t_compute = MPI_Wtime();
    }
#endif

    // ==========================================
    // --- 4) PURE MATHEMATICAL COMPUTATION ---
    // ==========================================
    vector<vector<Pair>> final_local_results(local_Q);

    // Compute everything without talking to the network
    for (int q_start = 0; q_start < local_Q; q_start += QBLOCK) {
        const int block_Q = min(local_Q - q_start, QBLOCK);
        vector<vector<Pair>> block_topk(block_Q);

        compute_topk(routine, kd_root.get(), A,
                    local_N, local_start,
                    block_Q, q_start,
                    local_flat_d, local_flat_q,
                    local_query_ks, block_topk); 

        // Store results immediately into the final vector
        for (int ql = 0; ql < block_Q; ql++) {
            final_local_results[q_start + ql] = block_topk[ql];
        }
    }

#ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        t_compute = MPI_Wtime() - t_compute;
        t_reduce = MPI_Wtime();
    }
#endif

    // ==========================================
    // --- 5) SINGLE NETWORK TREE REDUCTION ---
    // ==========================================
    // Calculate global offsets for packing the ENTIRE local_Q chunk
    vector<int> q_off(local_Q + 1, 0);
    for (int ql = 0; ql < local_Q; ql++) {
        q_off[ql + 1] = q_off[ql] + local_query_ks[ql];
    }
    const int pack_total = q_off[local_Q];

    // Allocate the unified reduction buffers
    vector<double> pack_d(pack_total, numeric_limits<double>::max());
    vector<int> pack_i(pack_total, -1);
    vector<double> recv_d(pack_total);
    vector<int> recv_i(pack_total);

    auto pack_all = [&]() {
        fill(pack_d.begin(), pack_d.end(), numeric_limits<double>::max());
        fill(pack_i.begin(), pack_i.end(), -1);
        for (int ql = 0; ql < local_Q; ql++) {
            int off = q_off[ql];
            for (int j = 0; j < (int)final_local_results[ql].size(); j++) {
                pack_d[off + j] = final_local_results[ql][j].first;
                pack_i[off + j] = final_local_results[ql][j].second;
            }
        }
    };

    auto unpack_to_all = [&](const vector<double>& rd, const vector<int>& ri) {
        for (int ql = 0; ql < local_Q; ql++) {
            int kq = local_query_ks[ql]; 
            int off = q_off[ql];
            vector<Pair> other;
            other.reserve(kq);
            for (int j = 0; j < kq; j++) {
                if (ri[off + j] != -1) {
                    other.push_back({rd[off + j], ri[off + j]});
                }
            }
            final_local_results[ql] = merge_topk_sorted(final_local_results[ql], other, kq);
        }
    };

    // Execute the Tree Reduction EXACTLY ONCE
    int step = 1;
    while (step < shared_query_size) {
        if (shared_query_rank % (2 * step) == 0) {
            int partner = shared_query_rank + step;
            if (partner < shared_query_size) {
                MPI_Recv(recv_d.data(), pack_total, MPI_DOUBLE, partner, 0, shared_query_comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_i.data(), pack_total, MPI_INT, partner, 1, shared_query_comm, MPI_STATUS_IGNORE);
                unpack_to_all(recv_d, recv_i);
            }
        } else {
            int partner = shared_query_rank - step;
            pack_all();
            MPI_Send(pack_d.data(), pack_total, MPI_DOUBLE, partner, 0, shared_query_comm);
            MPI_Send(pack_i.data(), pack_total, MPI_INT, partner, 1, shared_query_comm);
            break;
        }
        step *= 2;
    }
    
#ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        t_reduce = MPI_Wtime() - t_reduce;
        t_sendback = MPI_Wtime();
    }
    
    fprintf(stderr, "Reduce successful\n");
#endif

    // ==========================================
    // --- 6) FINALIZE RESULTS ON GLOBAL RANK 0 ---
    // ==========================================
    if (shared_query_rank == 0) {
        int total_k = 0;
        for (int q = 0; q < local_Q; q++) total_k += local_query_ks[q]; 

        vector<double> flat_d(total_k);
        vector<int> flat_i(total_k);
        int idx = 0;
        for (int q = 0; q < local_Q; q++) {
            for (auto& p : final_local_results[q]) {
                flat_d[idx] = p.first;
                flat_i[idx] = p.second;
                idx++;
            }
        }

        if (rank == 0) {
            // --- 5A. PRE-ALLOCATE AND POST NON-BLOCKING RECEIVES ---
            // Memory must stay alive and stable while background network transfers happen
            vector<vector<double>> recv_buffers_d(num_query_splits);
            vector<vector<int>> recv_buffers_i(num_query_splits);
            vector<MPI_Request> reqs_d(num_query_splits, MPI_REQUEST_NULL);
            vector<MPI_Request> reqs_i(num_query_splits, MPI_REQUEST_NULL);

            for (int qs = 1; qs < num_query_splits; qs++) {
                int src_rank = qs;
                if (!data_bcast_row) src_rank *= num_data_splits;
                int qs_Q = q_counts[qs];
                if (qs_Q == 0) continue;

                int qs_total_k = 0;
                for (int q = 0; q < qs_Q; q++) qs_total_k += query_ks[q_displs[qs] + q];

                recv_buffers_d[qs].resize(qs_total_k);
                recv_buffers_i[qs].resize(qs_total_k);

                // Post the requests. The network card starts working immediately.
                MPI_Irecv(recv_buffers_d[qs].data(), qs_total_k, MPI_DOUBLE, src_rank, 0, MPI_COMM_WORLD, &reqs_d[qs]);
                MPI_Irecv(recv_buffers_i[qs].data(), qs_total_k, MPI_INT, src_rank, 1, MPI_COMM_WORLD, &reqs_i[qs]);
            }

            // --- 5B. PROCESS LOCAL QUERIES (Overlaps with network transfer!) ---
            // While Rank 0 does slow I/O here, the remote data streams into the buffers above
            for (int q = 0; q < local_Q; q++) {
                int global_q = q_displs[0] + q;
                int best_lbl = get_best_label(final_local_results[q], dataset);
                reportResult(queries[global_q], final_local_results[q], best_lbl);
            }

            // --- 5C. WAIT AND PROCESS REMOTE QUERIES ---
            for (int qs = 1; qs < num_query_splits; qs++) {
                int qs_Q = q_counts[qs];
                if (qs_Q == 0) continue;

                // Safely block until this specific split's data is fully in RAM
                MPI_Wait(&reqs_d[qs], MPI_STATUS_IGNORE);
                MPI_Wait(&reqs_i[qs], MPI_STATUS_IGNORE);

                int recv_idx = 0;
                for (int q = 0; q < qs_Q; q++) {
                    int global_q = q_displs[qs] + q;
                    int kq = query_ks[global_q];
                    vector<Pair> res(kq);
                    for (int k = 0; k < kq; k++) {
                        res[k] = {recv_buffers_d[qs][recv_idx], recv_buffers_i[qs][recv_idx]};
                        recv_idx++;
                    }
                    int best_lbl = get_best_label(res, dataset);
                    reportResult(queries[global_q], res, best_lbl);
                }
            }
        } else {
            // Other roots send data back to Global Rank 0
            // Because Rank 0 already posted the Irecv, these Sends will clear the network instantly
            MPI_Send(flat_d.data(), total_k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(flat_i.data(), total_k, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef DEBUG
    // Output cleanly to standard error
    if (rank == 0) {
        t_sendback = MPI_Wtime() - t_sendback;
        fprintf(stderr, "\n--- MPI Rank 0 Performance Profile ---\n");
        fprintf(stderr, " 1. Precompute   : %10.4f s\n", t_pcomp);
        fprintf(stderr, " 2. Send Data    : %10.4f s\n", t_data);
        fprintf(stderr, " 3. Send Query   : %10.4f s\n", t_query);
        fprintf(stderr, " 4. Build Tree   : %10.4f s\n", t_build);
        fprintf(stderr, " 5. Computation  : %10.4f s\n", t_compute);
        fprintf(stderr, " 6. Reduce       : %10.4f s\n", t_reduce);
        fprintf(stderr, " 7. Sendback     : %10.4f s\n", t_sendback);
        fprintf(stderr, " -------------------------------------\n");
        fprintf(stderr, " Total Time      : %10.4f s\n\n", t_pcomp + t_data + t_query + t_build + t_compute + t_reduce + t_sendback);
    }
#endif

    // Clean up custom communicators
    MPI_Comm_free(&shared_data_comm);
    MPI_Comm_free(&shared_query_comm);
}