// Exact/heuristic prefilter for non-neat dimension-9 smooth polytopes.
//
// Input is the binary stream produced by polydb_binary_stream.py.  Each facet
// row is [1,a] with inequality 1+a.y >= 0.  Every representative contains the
// coordinate facets a=-e_i.  Translate a displacement so those nine support
// shifts are zero.  At the coordinate vertex q=(1,...,1), normal-fan
// preservation for both +/-s implies |s_j| <= a_j.q for every remaining
// facet.  For every integer s in this box, a possible middle lattice point is
// necessarily y in {-1,0,1}^9 and must satisfy |s_j-a_j.y| <= 1.
//
// Thus, if the union of these middle-point conditions covers the whole box,
// the polytope is neat without any further computation.  If an uncovered s is
// found, the record is emitted for the subsequent exact normal-fan check.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int D = 9;
constexpr int NX = 19683;
constexpr int WORDS = (NX + 63) / 64;

using Point = std::array<std::int8_t, D>;
using Normal = std::array<std::int16_t, D>;
using Mask = std::array<std::uint64_t, WORDS>;

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

struct ShiftKey {
    Normal normal{};
    std::int16_t shift = 0;
    bool operator==(const ShiftKey& other) const noexcept {
        return shift == other.shift && normal == other.normal;
    }
};

struct ShiftKeyHash {
    std::size_t operator()(const ShiftKey& key) const noexcept {
        std::uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (std::int16_t x : key.normal) {
            h ^= static_cast<std::uint64_t>(static_cast<std::int64_t>(x) + 32768)
                 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        h ^= static_cast<std::uint64_t>(static_cast<std::int64_t>(key.shift) + 32768)
             + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return static_cast<std::size_t>(h);
    }
};

struct Options {
    std::string output;
    std::string shard;
    std::string ranges;
    std::uint64_t exact_box_limit = 100000;
    int heuristic_trials = 500;
    int retain = 100;
};

Options parse_options(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value after " + arg);
            return argv[i];
        };
        if (arg == "--output") opts.output = value();
        else if (arg == "--shard") opts.shard = value();
        else if (arg == "--ranges") opts.ranges = value();
        else if (arg == "--exact-box-limit") opts.exact_box_limit = std::stoull(value());
        else if (arg == "--heuristic-trials") opts.heuristic_trials = std::stoi(value());
        else if (arg == "--retain") opts.retain = std::stoi(value());
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (opts.output.empty() || opts.shard.empty() || opts.ranges.empty()) {
        throw std::runtime_error("--output, --shard and --ranges are required");
    }
    if (opts.exact_box_limit == 0 || opts.heuristic_trials < 0 || opts.retain <= 0) {
        throw std::runtime_error("invalid numeric option");
    }
    return opts;
}

std::vector<Point> make_points() {
    std::vector<Point> result;
    result.reserve(NX);
    for (int code = 0; code < NX; ++code) {
        int t = code;
        Point p{};
        for (int j = D - 1; j >= 0; --j) {
            p[j] = static_cast<std::int8_t>((t % 3) - 1);
            t /= 3;
        }
        result.push_back(p);
    }
    return result;
}

Mask all_mask() {
    Mask m{};
    m.fill(~std::uint64_t{0});
    const int used = NX % 64;
    if (used) m.back() = (std::uint64_t{1} << used) - 1;
    return m;
}

int popcount(const Mask& m) {
    int result = 0;
    for (std::uint64_t word : m) result += __builtin_popcountll(word);
    return result;
}

bool empty(const Mask& m) {
    for (std::uint64_t word : m) if (word) return false;
    return true;
}

Mask intersect(const Mask& a, const Mask& b) {
    Mask result{};
    for (int i = 0; i < WORDS; ++i) result[i] = a[i] & b[i];
    return result;
}

Normal canonical_normal(Normal u, int& shift) {
    for (std::int16_t x : u) {
        if (x < 0) {
            for (auto& a : u) a = static_cast<std::int16_t>(-a);
            shift = -shift;
            return u;
        }
        if (x > 0) return u;
    }
    throw std::runtime_error("zero normal");
}

class ShiftMaskCache {
public:
    explicit ShiftMaskCache(const std::vector<Point>& points) : points_(points) {
        cache_.reserve(100000);
    }

    const Mask& get(const Normal& raw, int raw_shift) {
        if (raw_shift < -32767 || raw_shift > 32767) {
            throw std::runtime_error("support shift outside int16");
        }
        int shift = raw_shift;
        Normal normal = canonical_normal(raw, shift);
        ShiftKey key{normal, static_cast<std::int16_t>(shift)};
        auto it = cache_.find(key);
        if (it != cache_.end()) return it->second;
        Mask mask{};
        for (int i = 0; i < NX; ++i) {
            int dot = 0;
            for (int j = 0; j < D; ++j) {
                dot += static_cast<int>(normal[j]) * static_cast<int>(points_[i][j]);
            }
            if (dot >= shift - 1 && dot <= shift + 1) {
                mask[i >> 6] |= std::uint64_t{1} << (i & 63);
            }
        }
        return cache_.emplace(key, mask).first->second;
    }

    std::size_t size() const { return cache_.size(); }

private:
    const std::vector<Point>& points_;
    std::unordered_map<ShiftKey, Mask, ShiftKeyHash> cache_;
};

std::uint8_t read_u8(std::istream& in) {
    char b;
    if (!in.read(&b, 1)) throw std::runtime_error("unexpected end of stream");
    return static_cast<std::uint8_t>(static_cast<unsigned char>(b));
}

std::uint32_t read_u32(std::istream& in) {
    std::array<unsigned char, 4> b{};
    if (!in.read(reinterpret_cast<char*>(b.data()), 4)) throw std::runtime_error("unexpected end of stream");
    return static_cast<std::uint32_t>(b[0])
         | (static_cast<std::uint32_t>(b[1]) << 8)
         | (static_cast<std::uint32_t>(b[2]) << 16)
         | (static_cast<std::uint32_t>(b[3]) << 24);
}

std::uint64_t read_u64(std::istream& in) {
    std::array<unsigned char, 8> b{};
    if (!in.read(reinterpret_cast<char*>(b.data()), 8)) throw std::runtime_error("unexpected end of stream");
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(b[i]) << (8 * i);
    return value;
}

std::int16_t read_i16(std::istream& in) {
    std::array<unsigned char, 2> b{};
    if (!in.read(reinterpret_cast<char*>(b.data()), 2)) throw std::runtime_error("unexpected end of stream");
    const std::uint16_t u = static_cast<std::uint16_t>(b[0])
                          | (static_cast<std::uint16_t>(b[1]) << 8);
    return static_cast<std::int16_t>(u);
}

std::string poly_id(int n_facets, std::uint32_t serial) {
    std::ostringstream out;
    out << "F.9D.f" << n_facets << '.' << std::setw(7) << std::setfill('0') << serial;
    return out.str();
}

bool is_negative_coordinate(const Normal& a) {
    int minus = 0;
    for (std::int16_t x : a) {
        if (x == -1) ++minus;
        else if (x != 0) return false;
    }
    return minus == 1;
}

std::uint64_t capped_product(std::uint64_t a, std::uint64_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > std::numeric_limits<std::uint64_t>::max() / b) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return a * b;
}

struct Variable {
    int facet_index = -1;
    int bound = 0;
    std::vector<int> values;
    std::vector<const Mask*> masks;
    int smallest_mask = 0;
};

struct SearchResult {
    bool exact = false;
    bool uncovered = false;
    std::uint64_t assignments_tested = 0;
    int minimum_lifts = std::numeric_limits<int>::max();
    std::vector<int> best_shifts;
    std::vector<int> uncovered_shifts;
};

void exact_dfs(std::size_t depth,
               const std::vector<Variable>& variables,
               const Mask& current,
               std::vector<int>& assignment,
               SearchResult& result) {
    if (empty(current)) {
        result.uncovered = true;
        result.uncovered_shifts = assignment;
        for (std::size_t i = depth; i < variables.size(); ++i) {
            result.uncovered_shifts[i] = 0;
        }
        return;
    }
    if (depth == variables.size()) {
        ++result.assignments_tested;
        const int lifts = popcount(current);
        if (lifts < result.minimum_lifts) {
            result.minimum_lifts = lifts;
            result.best_shifts = assignment;
        }
        return;
    }
    const Variable& var = variables[depth];
    for (std::size_t i = 0; i < var.values.size(); ++i) {
        assignment[depth] = var.values[i];
        const Mask next = intersect(current, *var.masks[i]);
        exact_dfs(depth + 1, variables, next, assignment, result);
        if (result.uncovered) return;
    }
}

SearchResult search_box(const std::vector<Normal>& normals,
                        const std::vector<int>& extra_indices,
                        const std::vector<int>& bounds,
                        std::uint64_t box_volume,
                        std::uint64_t exact_limit,
                        int heuristic_trials,
                        std::uint64_t seed,
                        ShiftMaskCache& cache,
                        const Mask& all) {
    SearchResult result;
    Mask fixed = all;
    std::vector<Variable> variables;
    for (std::size_t k = 0; k < extra_indices.size(); ++k) {
        const int facet = extra_indices[k];
        const int b = bounds[k];
        if (b == 0) {
            fixed = intersect(fixed, cache.get(normals[facet], 0));
            continue;
        }
        Variable variable;
        variable.facet_index = facet;
        variable.bound = b;
        // More restrictive extreme values first, then work inward.
        for (int radius = b; radius >= 0; --radius) {
            if (radius == 0) {
                variable.values.push_back(0);
            } else {
                variable.values.push_back(-radius);
                variable.values.push_back(radius);
            }
        }
        variable.smallest_mask = NX;
        for (int value : variable.values) {
            const Mask& mask = cache.get(normals[facet], value);
            variable.masks.push_back(&mask);
            variable.smallest_mask = std::min(variable.smallest_mask, popcount(mask));
        }
        variables.push_back(std::move(variable));
    }
    std::sort(variables.begin(), variables.end(), [](const Variable& a, const Variable& b) {
        if (a.smallest_mask != b.smallest_mask) return a.smallest_mask < b.smallest_mask;
        if (a.bound != b.bound) return a.bound > b.bound;
        return a.facet_index < b.facet_index;
    });

    std::vector<int> assignment(variables.size(), 0);
    if (box_volume <= exact_limit) {
        result.exact = true;
        exact_dfs(0, variables, fixed, assignment, result);
        if (variables.empty()) {
            result.assignments_tested = 1;
            result.minimum_lifts = popcount(fixed);
            result.best_shifts.clear();
        }
        return result;
    }

    // Deterministic heuristic: greedy minimization with perturbed variable and
    // value orders.  Any empty intersection found is an exact no-lift witness;
    // failure to find one is explicitly not a proof of coverage.
    SplitMix64 rng(seed);
    result.minimum_lifts = popcount(fixed);
    result.best_shifts.assign(variables.size(), 0);
    const int trials = std::max(1, heuristic_trials);
    for (int trial = 0; trial < trials; ++trial) {
        std::vector<int> order(variables.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
        if (trial > 0) {
            for (std::size_t i = order.size(); i > 1; --i) {
                std::swap(order[i - 1], order[rng.next() % i]);
            }
        }
        Mask current = fixed;
        std::vector<int> chosen(variables.size(), 0);
        for (int index : order) {
            const Variable& var = variables[index];
            int best_value = 0;
            Mask best_mask{};
            int best_count = NX + 1;
            for (std::size_t k = 0; k < var.values.size(); ++k) {
                const Mask candidate = intersect(current, *var.masks[k]);
                const int count = popcount(candidate);
                if (count < best_count || (count == best_count && (rng.next() & 1))) {
                    best_count = count;
                    best_value = var.values[k];
                    best_mask = candidate;
                }
            }
            chosen[index] = best_value;
            current = best_mask;
            if (best_count == 0) {
                result.uncovered = true;
                result.uncovered_shifts = chosen;
                return result;
            }
        }
        ++result.assignments_tested;
        const int lifts = popcount(current);
        if (lifts < result.minimum_lifts) {
            result.minimum_lifts = lifts;
            result.best_shifts = chosen;
        }
    }
    return result;
}

struct Stored {
    int n_facets = 0;
    std::uint32_t serial = 0;
    std::uint64_t box_volume = 0;
    bool exact = false;
    bool uncovered = false;
    std::uint64_t assignments_tested = 0;
    int minimum_lifts = 0;
    std::vector<Normal> normals;
    std::vector<int> facet_shifts;
    std::vector<int> bounds;
};

struct RetainOrder {
    bool operator()(const Stored& a, const Stored& b) const {
        // Priority queue top is the least interesting retained record:
        // larger minimum-lift count, then smaller displacement box.
        if (a.minimum_lifts != b.minimum_lifts) return a.minimum_lifts < b.minimum_lifts;
        if (a.box_volume != b.box_volume) return a.box_volume > b.box_volume;
        if (a.n_facets != b.n_facets) return a.n_facets > b.n_facets;
        return a.serial > b.serial;
    }
};

std::string normal_json(const Normal& u) {
    std::ostringstream out;
    out << "[1";
    for (std::int16_t a : u) out << ',' << a;
    out << ']';
    return out.str();
}

void write_record(std::ostream& out, const Stored& s) {
    out << "{\"id\":\"" << poly_id(s.n_facets, s.serial) << "\""
        << ",\"n_facets\":" << s.n_facets
        << ",\"box_volume\":" << s.box_volume
        << ",\"exact_box_search\":" << (s.exact ? "true" : "false")
        << ",\"uncovered\":" << (s.uncovered ? "true" : "false")
        << ",\"assignments_tested\":" << s.assignments_tested
        << ",\"minimum_lifts\":" << s.minimum_lifts
        << ",\"bounds\":[";
    for (std::size_t i = 0; i < s.bounds.size(); ++i) {
        if (i) out << ',';
        out << s.bounds[i];
    }
    out << "]"
        << ",\"normalized_facet_shifts\":[";
    for (std::size_t i = 0; i < s.facet_shifts.size(); ++i) {
        if (i) out << ',';
        out << s.facet_shifts[i];
    }
    out << "]"
        << ",\"facets\":[";
    for (std::size_t i = 0; i < s.normals.size(); ++i) {
        if (i) out << ',';
        out << normal_json(s.normals[i]);
    }
    out << "]}";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_options(argc, argv);
        char magic[8]{};
        if (!std::cin.read(magic, 8)) throw std::runtime_error("missing stream header");
        const char expected_magic[8] = {'E','W','A','L','D','9','\0','\0'};
        if (std::memcmp(magic, expected_magic, 8) != 0) throw std::runtime_error("bad stream magic");
        const std::uint64_t expected_records = read_u64(std::cin);

        const std::vector<Point> points = make_points();
        const Mask all = all_mask();
        ShiftMaskCache cache(points);
        std::map<std::string, std::uint64_t> box_histogram;
        std::map<int, std::uint64_t> min_lift_histogram;
        std::priority_queue<Stored, std::vector<Stored>, RetainOrder> retained;
        std::vector<Stored> uncovered;
        std::uint64_t exact_records = 0;
        std::uint64_t heuristic_records = 0;
        std::uint64_t scanned = 0;
        std::uint64_t max_box = 0;
        int global_min_lifts = NX;

        for (; scanned < expected_records; ++scanned) {
            const int n_facets = read_u8(std::cin);
            const std::uint32_t serial = read_u32(std::cin);
            std::vector<Normal> normals(static_cast<std::size_t>(n_facets));
            for (int i = 0; i < n_facets; ++i) {
                for (int j = 0; j < D; ++j) normals[i][j] = read_i16(std::cin);
            }

            std::vector<int> extra_indices;
            std::vector<int> bounds;
            int coordinate_count = 0;
            std::uint64_t box_volume = 1;
            for (int i = 0; i < n_facets; ++i) {
                if (is_negative_coordinate(normals[i])) {
                    ++coordinate_count;
                    continue;
                }
                int bound = 0;
                for (std::int16_t a : normals[i]) bound += a;
                if (bound < 0) {
                    throw std::runtime_error("negative coordinate-vertex slack at " + poly_id(n_facets, serial));
                }
                extra_indices.push_back(i);
                bounds.push_back(bound);
                box_volume = capped_product(box_volume, static_cast<std::uint64_t>(2 * bound + 1));
            }
            if (coordinate_count != D || static_cast<int>(extra_indices.size()) != n_facets - D) {
                throw std::runtime_error("coordinate facet normalization failure at " + poly_id(n_facets, serial));
            }
            max_box = std::max(max_box, box_volume);
            std::string box_key = box_volume == std::numeric_limits<std::uint64_t>::max()
                ? "overflow" : std::to_string(box_volume);
            ++box_histogram[box_key];

            const std::uint64_t seed = 0x243f6a8885a308d3ULL
                ^ (static_cast<std::uint64_t>(n_facets) << 48)
                ^ (static_cast<std::uint64_t>(serial) * 0x9e3779b97f4a7c15ULL);
            SearchResult search = search_box(
                normals, extra_indices, bounds, box_volume,
                opts.exact_box_limit, opts.heuristic_trials, seed, cache, all);
            if (search.exact) ++exact_records;
            else ++heuristic_records;
            if (search.minimum_lifts == std::numeric_limits<int>::max()) search.minimum_lifts = 0;
            global_min_lifts = std::min(global_min_lifts, search.minimum_lifts);
            ++min_lift_histogram[search.minimum_lifts];

            Stored record;
            record.n_facets = n_facets;
            record.serial = serial;
            record.box_volume = box_volume;
            record.exact = search.exact;
            record.uncovered = search.uncovered;
            record.assignments_tested = search.assignments_tested;
            record.minimum_lifts = search.uncovered ? 0 : search.minimum_lifts;
            record.bounds = bounds;
            record.normals = normals;
            record.facet_shifts.assign(static_cast<std::size_t>(n_facets), 0);
            const std::vector<int>& chosen = search.uncovered ? search.uncovered_shifts : search.best_shifts;

            // Reconstruct the variable order used by search_box so chosen shifts
            // are assigned to the corresponding original facet indices.
            std::vector<Variable> variables;
            for (std::size_t k = 0; k < extra_indices.size(); ++k) {
                if (bounds[k] == 0) continue;
                Variable v;
                v.facet_index = extra_indices[k];
                v.bound = bounds[k];
                for (int radius = v.bound; radius >= 0; --radius) {
                    if (radius == 0) v.values.push_back(0);
                    else { v.values.push_back(-radius); v.values.push_back(radius); }
                }
                v.smallest_mask = NX;
                for (int value : v.values) {
                    const Mask& mask = cache.get(normals[v.facet_index], value);
                    v.masks.push_back(&mask);
                    v.smallest_mask = std::min(v.smallest_mask, popcount(mask));
                }
                variables.push_back(std::move(v));
            }
            std::sort(variables.begin(), variables.end(), [](const Variable& a, const Variable& b) {
                if (a.smallest_mask != b.smallest_mask) return a.smallest_mask < b.smallest_mask;
                if (a.bound != b.bound) return a.bound > b.bound;
                return a.facet_index < b.facet_index;
            });
            for (std::size_t i = 0; i < chosen.size() && i < variables.size(); ++i) {
                record.facet_shifts[variables[i].facet_index] = chosen[i];
            }

            if (search.uncovered) uncovered.push_back(record);
            retained.push(record);
            if (static_cast<int>(retained.size()) > opts.retain) retained.pop();

            if ((scanned + 1) % 100000 == 0) {
                std::cerr << opts.shard << ": neat-prefilter scanned=" << (scanned + 1)
                          << '/' << expected_records
                          << " exact=" << exact_records
                          << " uncovered=" << uncovered.size()
                          << " min_lifts=" << global_min_lifts
                          << " cache=" << cache.size() << '\n';
            }
        }
        char extra;
        if (std::cin.read(&extra, 1)) throw std::runtime_error("trailing bytes after expected records");

        std::vector<Stored> best;
        while (!retained.empty()) {
            best.push_back(retained.top());
            retained.pop();
        }
        std::sort(best.begin(), best.end(), [](const Stored& a, const Stored& b) {
            if (a.minimum_lifts != b.minimum_lifts) return a.minimum_lifts < b.minimum_lifts;
            if (a.box_volume != b.box_volume) return a.box_volume > b.box_volume;
            if (a.n_facets != b.n_facets) return a.n_facets < b.n_facets;
            return a.serial < b.serial;
        });

        std::ofstream out(opts.output);
        if (!out) throw std::runtime_error("cannot open output");
        out << "{\n"
            << "  \"dimension\":9,\n"
            << "  \"shard\":\"" << opts.shard << "\",\n"
            << "  \"ranges\":\"" << opts.ranges << "\",\n"
            << "  \"complete\":true,\n"
            << "  \"polytopes_scanned\":" << scanned << ",\n"
            << "  \"exact_box_limit\":" << opts.exact_box_limit << ",\n"
            << "  \"exact_records\":" << exact_records << ",\n"
            << "  \"heuristic_records\":" << heuristic_records << ",\n"
            << "  \"uncovered_count\":" << uncovered.size() << ",\n"
            << "  \"minimum_lifts_seen\":" << global_min_lifts << ",\n"
            << "  \"maximum_box_volume\":" << max_box << ",\n"
            << "  \"cached_shift_constraints\":" << cache.size() << ",\n";

        out << "  \"box_volume_histogram\":{";
        bool first = true;
        for (const auto& [key, count] : box_histogram) {
            if (!first) out << ',';
            first = false;
            out << "\n    \"" << key << "\":" << count;
        }
        if (!box_histogram.empty()) out << '\n';
        out << "  },\n";

        out << "  \"minimum_lift_histogram\":{";
        first = true;
        for (const auto& [key, count] : min_lift_histogram) {
            if (!first) out << ',';
            first = false;
            out << "\n    \"" << key << "\":" << count;
        }
        if (!min_lift_histogram.empty()) out << '\n';
        out << "  },\n";

        out << "  \"best_records\":[";
        for (std::size_t i = 0; i < best.size(); ++i) {
            if (i) out << ',';
            out << "\n    ";
            write_record(out, best[i]);
        }
        if (!best.empty()) out << '\n';
        out << "  ],\n";

        out << "  \"uncovered\":[";
        for (std::size_t i = 0; i < uncovered.size(); ++i) {
            if (i) out << ',';
            out << "\n    ";
            write_record(out, uncovered[i]);
        }
        if (!uncovered.empty()) out << '\n';
        out << "  ]\n}\n";

        std::cerr << opts.shard << ": neat-prefilter complete scanned=" << scanned
                  << " exact=" << exact_records
                  << " heuristic=" << heuristic_records
                  << " uncovered=" << uncovered.size()
                  << " min_lifts=" << global_min_lifts
                  << " max_box=" << max_box << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << '\n';
        return 2;
    }
}
