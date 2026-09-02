// Independent exact Ewald-basis scan for the dimension-8 smooth-Fano database.
//
// Database ray generators u define the symmetric lattice-point set of the
// polar monotone polytope by |u.x| <= 1.  The first eight rays are e_i, hence
// every Ewald point lies in {-1,0,1}^8.  Membership uses fixed 6561-bit masks;
// every reported basis determinant is computed by fraction-free integer
// elimination.  Randomness only orders the search and is deterministically
// seeded by the database ID.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::cpp_int;

namespace {

constexpr int D = 8;
constexpr int NX = 6561;                  // 3^8
constexpr int WORDS = (NX + 63) / 64;    // 103
constexpr int BLOCK_SIZE = 7498;
constexpr int MAX_TRIALS = 5000;

using Vec = std::array<int, D>;
using Mask = std::array<std::uint64_t, WORDS>;

struct VecHash {
    std::size_t operator()(const Vec& v) const noexcept {
        std::uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (int x : v) {
            h ^= static_cast<std::uint64_t>(x + 32) + 0x9e3779b97f4a7c15ULL
                 + (h << 6) + (h >> 2);
        }
        return static_cast<std::size_t>(h);
    }
};

struct SplitMix64 {
    std::uint64_t state;
    explicit SplitMix64(std::uint64_t seed) : state(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

std::vector<Vec> make_points() {
    std::vector<Vec> points;
    points.reserve(NX);
    for (int code = 0; code < NX; ++code) {
        int t = code;
        Vec x{};
        for (int j = D - 1; j >= 0; --j) {
            x[j] = (t % 3) - 1;
            t /= 3;
        }
        points.push_back(x);
    }
    return points;
}

Mask all_mask() {
    Mask m{};
    m.fill(~std::uint64_t{0});
    const int used = NX % 64;
    if (used) m.back() = (std::uint64_t{1} << used) - 1;
    return m;
}

bool contains(const Mask& m, int index) {
    return (m[index >> 6] >> (index & 63)) & std::uint64_t{1};
}

int popcount(const Mask& m) {
    int n = 0;
    for (std::uint64_t w : m) n += __builtin_popcountll(w);
    return n;
}

std::vector<int> integer_to_sequence(cpp_int n, int base) {
    if (n < 0 || base < 2) throw std::runtime_error("invalid encoding");
    std::vector<int> digits;
    while (n != 0) {
        cpp_int q = n / base;
        cpp_int r = n - q * base;
        digits.push_back(r.convert_to<int>());
        n = q;
    }
    if (digits.empty()) digits.push_back(0);
    return digits;
}

std::vector<Vec> decode(const std::string& line, int base) {
    cpp_int n(line);
    const std::vector<int> digits = integer_to_sequence(n, base);
    if (digits.size() < 2 || digits[0] != D) {
        throw std::runtime_error("bad dimension in encoded record");
    }
    const int shift = digits[1];
    const std::size_t ncoeff = digits.size() - 2;
    if (ncoeff % D) throw std::runtime_error("bad coefficient count");
    std::vector<Vec> vertices(ncoeff / D);
    for (std::size_t i = 0; i < ncoeff; ++i) {
        vertices[i / D][i % D] = digits[i + 2] - shift;
    }
    if (vertices.size() < D) throw std::runtime_error("too few rays");
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            if (vertices[i][j] != (i == j ? 1 : 0)) {
                throw std::runtime_error("representative lacks initial standard facet");
            }
        }
    }
    return vertices;
}

class RayMaskCache {
public:
    explicit RayMaskCache(const std::vector<Vec>& points) : points_(points) {}

    const Mask& get(const Vec& u) {
        auto it = cache_.find(u);
        if (it != cache_.end()) return it->second;
        Mask m{};
        for (int i = 0; i < NX; ++i) {
            int dot = 0;
            for (int j = 0; j < D; ++j) dot += u[j] * points_[i][j];
            if (dot >= -1 && dot <= 1) {
                m[i >> 6] |= std::uint64_t{1} << (i & 63);
            }
        }
        return cache_.emplace(u, m).first->second;
    }

    std::size_t size() const { return cache_.size(); }

private:
    const std::vector<Vec>& points_;
    std::unordered_map<Vec, Mask, VecHash> cache_;
};

Mask ewald_mask(const std::vector<Vec>& vertices, RayMaskCache& cache, const Mask& all) {
    Mask result = all;
    for (std::size_t i = D; i < vertices.size(); ++i) {
        const Mask& constraint = cache.get(vertices[i]);
        for (int w = 0; w < WORDS; ++w) result[w] &= constraint[w];
    }
    return result;
}

long long determinant(const std::array<int, D>& point_ids,
                      const std::vector<Vec>& points) {
    long long a[D][D]{};
    for (int col = 0; col < D; ++col) {
        const Vec& v = points[point_ids[col]];
        for (int row = 0; row < D; ++row) a[row][col] = v[row];
    }

    long long previous = 1;
    int sign = 1;
    for (int k = 0; k < D - 1; ++k) {
        int pivot_row = k;
        while (pivot_row < D && a[pivot_row][k] == 0) ++pivot_row;
        if (pivot_row == D) return 0;
        if (pivot_row != k) {
            for (int j = 0; j < D; ++j) std::swap(a[k][j], a[pivot_row][j]);
            sign = -sign;
        }
        const long long pivot = a[k][k];
        for (int i = k + 1; i < D; ++i) {
            for (int j = k + 1; j < D; ++j) {
                const long long numerator = a[i][j] * pivot - a[i][k] * a[k][j];
                if (numerator % previous != 0) {
                    throw std::runtime_error("non-exact Bareiss division");
                }
                a[i][j] = numerator / previous;
            }
        }
        previous = pivot;
        for (int i = k + 1; i < D; ++i) a[i][k] = 0;
    }
    return static_cast<long long>(sign) * a[D - 1][D - 1];
}

struct BasisResult {
    bool found = false;
    int trials = 0;
    std::array<int, D> point_ids{};
};

BasisResult find_basis(const Mask& ewald,
                       std::uint64_t poly_id,
                       const std::vector<int>& representatives,
                       const std::vector<Vec>& points) {
    SplitMix64 rng(0xd1b54a32d192ed03ULL ^ (poly_id * 0x9e3779b97f4a7c15ULL));
    BasisResult result;

    for (int trial = 1; trial <= MAX_TRIALS; ++trial) {
        std::array<int, D> chosen{};
        int got = 0;
        int draws = 0;
        while (got < D && draws < 200000) {
            ++draws;
            const int id = representatives[rng.next() % representatives.size()];
            if (!contains(ewald, id)) continue;
            bool duplicate = false;
            for (int i = 0; i < got; ++i) duplicate |= (chosen[i] == id);
            if (!duplicate) chosen[got++] = id;
        }
        if (got < D) throw std::runtime_error("failed to sample eight Ewald representatives");
        const long long det = determinant(chosen, points);
        if (det == 1 || det == -1) {
            result.found = true;
            result.trials = trial;
            result.point_ids = chosen;
            return result;
        }
    }
    result.trials = MAX_TRIALS;
    return result;
}

std::string vec_json(const Vec& v) {
    std::ostringstream out;
    out << '[';
    for (int i = 0; i < D; ++i) {
        if (i) out << ',';
        out << v[i];
    }
    out << ']';
    return out.str();
}

struct Unresolved {
    int id;
    int ewald_count;
    std::vector<Vec> vertices;
};

struct Worst {
    int trials;
    int id;
    int ewald_count;
    std::array<int, D> basis;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: ewald_basis_scan DATA_DIR OUTPUT_JSON\n";
        return 2;
    }
    const std::string data_dir = argv[1];
    const std::string output_path = argv[2];

    const auto started = std::chrono::steady_clock::now();
    const std::vector<Vec> points = make_points();
    const Mask all = all_mask();
    std::vector<int> representatives;
    representatives.reserve((NX - 1) / 2);
    int zero_index = -1;
    for (int i = 0; i < NX; ++i) {
        const Vec& v = points[i];
        int first = 0;
        for (int x : v) {
            if (x != 0) { first = x; break; }
        }
        if (first > 0) representatives.push_back(i);
        if (first == 0) zero_index = i;
    }
    if (zero_index < 0 || representatives.size() != 3280) {
        throw std::runtime_error("bad candidate universe");
    }

    RayMaskCache cache(points);
    std::map<int, long long> trials_histogram;
    std::vector<Unresolved> unresolved;
    std::vector<Worst> worst;
    long long scanned = 0;
    long long bases_found = 0;
    int minimum_ewald = std::numeric_limits<int>::max();
    int maximum_trials = 0;

    for (int block = 0; block <= 100; ++block) {
        std::ifstream in(data_dir + "/block" + std::to_string(block));
        if (!in) throw std::runtime_error("cannot open block " + std::to_string(block));
        std::string line;
        if (!std::getline(in, line)) throw std::runtime_error("empty block");
        const int base = std::stoi(line);
        int offset = 0;
        int block_unresolved = 0;
        int block_max_trials = 0;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            ++offset;
            const int id = block * BLOCK_SIZE + offset;
            const std::vector<Vec> vertices = decode(line, base);
            const Mask ewald = ewald_mask(vertices, cache, all);
            if (!contains(ewald, zero_index)) throw std::runtime_error("zero absent");
            const int n_ewald = popcount(ewald);
            if ((n_ewald & 1) == 0) throw std::runtime_error("non-symmetric Ewald count");
            minimum_ewald = std::min(minimum_ewald, n_ewald);

            const BasisResult basis = find_basis(ewald, id, representatives, points);
            ++scanned;
            ++trials_histogram[basis.trials];
            maximum_trials = std::max(maximum_trials, basis.trials);
            block_max_trials = std::max(block_max_trials, basis.trials);
            if (basis.found) {
                ++bases_found;
                Worst w{basis.trials, id, n_ewald, basis.point_ids};
                worst.push_back(w);
                std::sort(worst.begin(), worst.end(), [](const Worst& a, const Worst& b) {
                    if (a.trials != b.trials) return a.trials > b.trials;
                    return a.id < b.id;
                });
                if (worst.size() > 100) worst.resize(100);
            } else {
                ++block_unresolved;
                unresolved.push_back(Unresolved{id, n_ewald, vertices});
            }
        }
        std::cout << "block " << std::setw(3) << block
                  << ": records=" << offset
                  << ", max trials=" << block_max_trials
                  << ", unresolved=" << block_unresolved
                  << ", cached rays=" << cache.size() << '\n';
    }

    if (scanned != 749892) {
        throw std::runtime_error("classification record count mismatch: " + std::to_string(scanned));
    }

    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::ofstream out(output_path);
    if (!out) throw std::runtime_error("cannot create output");
    out << "{\n";
    out << "  \"dimension\": 8,\n";
    out << "  \"candidate_box\": [[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]],\n";
    out << "  \"polytopes_scanned\": " << scanned << ",\n";
    out << "  \"bases_found\": " << bases_found << ",\n";
    out << "  \"unresolved_count\": " << unresolved.size() << ",\n";
    out << "  \"minimum_ewald_points\": " << minimum_ewald << ",\n";
    out << "  \"distinct_ray_constraints\": " << cache.size() << ",\n";
    out << "  \"maximum_trials\": " << maximum_trials << ",\n";
    out << "  \"max_trials_per_polytope\": " << MAX_TRIALS << ",\n";
    out << "  \"elapsed_seconds\": " << std::setprecision(10) << seconds << ",\n";

    out << "  \"trials_histogram\": {";
    bool first = true;
    for (const auto& [trials, count] : trials_histogram) {
        if (!first) out << ',';
        first = false;
        out << "\n    \"" << trials << "\": " << count;
    }
    if (!trials_histogram.empty()) out << '\n';
    out << "  },\n";

    out << "  \"worst_resolved\": [";
    for (std::size_t i = 0; i < worst.size(); ++i) {
        if (i) out << ',';
        const Worst& w = worst[i];
        out << "\n    {\"id\":" << w.id
            << ",\"n_ewald_points\":" << w.ewald_count
            << ",\"trials\":" << w.trials
            << ",\"basis\":[";
        for (int j = 0; j < D; ++j) {
            if (j) out << ',';
            out << vec_json(points[w.basis[j]]);
        }
        out << "]}";
    }
    if (!worst.empty()) out << '\n';
    out << "  ],\n";

    out << "  \"unresolved\": [";
    for (std::size_t i = 0; i < unresolved.size(); ++i) {
        if (i) out << ',';
        const Unresolved& u = unresolved[i];
        out << "\n    {\"id\":" << u.id
            << ",\"n_ewald_points\":" << u.ewald_count
            << ",\"vertices\":[";
        for (std::size_t j = 0; j < u.vertices.size(); ++j) {
            if (j) out << ',';
            out << vec_json(u.vertices[j]);
        }
        out << "]}";
    }
    if (!unresolved.empty()) out << '\n';
    out << "  ]\n";
    out << "}\n";

    std::cout << "scanned=" << scanned
              << " bases=" << bases_found
              << " unresolved=" << unresolved.size()
              << " max_trials=" << maximum_trials
              << " seconds=" << seconds << '\n';
    return unresolved.empty() ? 0 : 3;
}
