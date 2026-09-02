// Exact dimension-9 Ewald scanner backend.
// Binary input (little-endian): magic[8]="EWALD9\0\0", uint64 record count,
// then for each record: uint8 N_FACETS, uint32 serial, and N_FACETS*9 int16
// facet-normal coordinates.  The corresponding facet inequalities are
// 1 + a.x >= 0.  Thus x is an Ewald point iff |a.x| <= 1 for every normal a.

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
constexpr int NX = 19683;                 // 3^9
constexpr int WORDS = (NX + 63) / 64;    // 308

using Point = std::array<std::int8_t, D>;
using Normal = std::array<std::int16_t, D>;
using Mask = std::array<std::uint64_t, WORDS>;

struct NormalHash {
    std::size_t operator()(const Normal& v) const noexcept {
        std::uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (std::int16_t x : v) {
            h ^= static_cast<std::uint64_t>(static_cast<std::int64_t>(x) + 32768)
                 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
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

struct Options {
    std::string output;
    std::string shard;
    std::string ranges;
    int top_k = 100;
    int max_trials = 5000;
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
        else if (arg == "--top-k") opts.top_k = std::stoi(value());
        else if (arg == "--max-trials") opts.max_trials = std::stoi(value());
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (opts.output.empty() || opts.shard.empty() || opts.ranges.empty()) {
        throw std::runtime_error("--output, --shard, and --ranges are required");
    }
    if (opts.top_k <= 0 || opts.max_trials <= 0) {
        throw std::runtime_error("top-k and max-trials must be positive");
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

bool contains(const Mask& m, int index) {
    return ((m[index >> 6] >> (index & 63)) & std::uint64_t{1}) != 0;
}

int popcount(const Mask& m) {
    int result = 0;
    for (std::uint64_t word : m) result += __builtin_popcountll(word);
    return result;
}

Normal canonical(Normal u) {
    for (std::int16_t x : u) {
        if (x < 0) {
            for (auto& a : u) a = static_cast<std::int16_t>(-a);
            return u;
        }
        if (x > 0) return u;
    }
    throw std::runtime_error("zero facet normal");
}

class RayMaskCache {
public:
    explicit RayMaskCache(const std::vector<Point>& points) : points_(points) {
        cache_.reserve(50000);
    }

    const Mask& get(const Normal& raw) {
        const Normal u = canonical(raw);
        auto it = cache_.find(u);
        if (it != cache_.end()) return it->second;
        Mask m{};
        for (int i = 0; i < NX; ++i) {
            int dot = 0;
            for (int j = 0; j < D; ++j) {
                dot += static_cast<int>(u[j]) * static_cast<int>(points_[i][j]);
            }
            if (dot >= -1 && dot <= 1) {
                m[i >> 6] |= std::uint64_t{1} << (i & 63);
            }
        }
        return cache_.emplace(u, m).first->second;
    }

    std::size_t size() const { return cache_.size(); }

private:
    const std::vector<Point>& points_;
    std::unordered_map<Normal, Mask, NormalHash> cache_;
};

std::uint8_t read_u8(std::istream& in) {
    char b;
    if (!in.read(&b, 1)) throw std::runtime_error("unexpected end of binary stream");
    return static_cast<std::uint8_t>(static_cast<unsigned char>(b));
}

std::uint32_t read_u32(std::istream& in) {
    std::array<unsigned char, 4> b{};
    if (!in.read(reinterpret_cast<char*>(b.data()), 4)) {
        throw std::runtime_error("unexpected end of binary stream");
    }
    return static_cast<std::uint32_t>(b[0])
         | (static_cast<std::uint32_t>(b[1]) << 8)
         | (static_cast<std::uint32_t>(b[2]) << 16)
         | (static_cast<std::uint32_t>(b[3]) << 24);
}

std::uint64_t read_u64(std::istream& in) {
    std::array<unsigned char, 8> b{};
    if (!in.read(reinterpret_cast<char*>(b.data()), 8)) {
        throw std::runtime_error("unexpected end of binary stream");
    }
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(b[i]) << (8 * i);
    return value;
}

std::int16_t read_i16(std::istream& in) {
    std::array<unsigned char, 2> b{};
    if (!in.read(reinterpret_cast<char*>(b.data()), 2)) {
        throw std::runtime_error("unexpected end of binary stream");
    }
    const std::uint16_t u = static_cast<std::uint16_t>(b[0])
                          | (static_cast<std::uint16_t>(b[1]) << 8);
    return static_cast<std::int16_t>(u);
}

std::string poly_id(int n_facets, std::uint32_t serial) {
    std::ostringstream out;
    out << "F.9D.f" << n_facets << '.' << std::setw(7) << std::setfill('0') << serial;
    return out.str();
}

long long determinant(const std::array<int, D>& ids, const std::vector<Point>& points) {
    long long a[D][D]{};
    for (int col = 0; col < D; ++col) {
        for (int row = 0; row < D; ++row) {
            a[row][col] = points[ids[col]][row];
        }
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

struct GF2Result {
    int rank = 0;
    std::array<int, D> chosen{};
};

GF2Result gf2_basis(const std::vector<int>& available, const std::vector<int>& parity) {
    std::array<int, D> pivots{};
    GF2Result result;
    for (int id : available) {
        int v = parity[id];
        if (v == 0) continue;
        for (int bit = D - 1; bit >= 0; --bit) {
            if (((v >> bit) & 1) == 0) continue;
            if (pivots[bit] != 0) {
                v ^= pivots[bit];
            } else {
                pivots[bit] = v;
                result.chosen[result.rank++] = id;
                break;
            }
        }
        if (result.rank == D) break;
    }
    return result;
}

struct BasisResult {
    bool found = false;
    int trials = 0;
    long long sampled_gcd = 0;
    long long minimum_nonzero_abs_det = 0;
    std::array<int, D> ids{};
};

long long gcdll(long long a, long long b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) {
        const long long r = a % b;
        a = b;
        b = r;
    }
    return a;
}

BasisResult find_basis(const GF2Result& gf2,
                       const std::vector<int>& available,
                       std::uint64_t seed,
                       int max_trials,
                       const std::vector<Point>& points) {
    BasisResult result;
    if (gf2.rank < D || static_cast<int>(available.size()) < D) return result;

    auto register_det = [&](long long det, const std::array<int, D>& ids, int trial) -> bool {
        if (det != 0) {
            const long long ad = det < 0 ? -det : det;
            result.sampled_gcd = gcdll(result.sampled_gcd, ad);
            if (result.minimum_nonzero_abs_det == 0 || ad < result.minimum_nonzero_abs_det) {
                result.minimum_nonzero_abs_det = ad;
            }
            if (ad == 1) {
                result.found = true;
                result.trials = trial;
                result.ids = ids;
                return true;
            }
        }
        return false;
    };

    // The parity-independent set has odd determinant, and is often already unimodular.
    if (register_det(determinant(gf2.chosen, points), gf2.chosen, 1)) return result;

    SplitMix64 rng(seed);
    for (int trial = 2; trial <= max_trials; ++trial) {
        std::array<int, D> chosen{};
        int got = 0;
        while (got < D) {
            const int id = available[rng.next() % available.size()];
            bool duplicate = false;
            for (int j = 0; j < got; ++j) duplicate |= chosen[j] == id;
            if (!duplicate) chosen[got++] = id;
        }
        if (register_det(determinant(chosen, points), chosen, trial)) return result;
    }
    result.trials = max_trials;
    return result;
}

std::string point_json(const Point& p) {
    std::ostringstream out;
    out << '[';
    for (int i = 0; i < D; ++i) {
        if (i) out << ',';
        out << static_cast<int>(p[i]);
    }
    out << ']';
    return out.str();
}

std::string normal_json(const Normal& u) {
    std::ostringstream out;
    out << "[1";
    for (int i = 0; i < D; ++i) out << ',' << u[i];
    out << ']';
    return out.str();
}

struct Stored {
    int n_facets = 0;
    std::uint32_t serial = 0;
    int ewald_count = 0;
    int gf2_rank = 0;
    Mask mask{};
    std::vector<Normal> normals;
    BasisResult basis;

    bool operator<(const Stored& other) const {
        if (ewald_count != other.ewald_count) return ewald_count < other.ewald_count;
        if (n_facets != other.n_facets) return n_facets < other.n_facets;
        return serial < other.serial;
    }
};

struct Worst {
    int trials = 0;
    int n_facets = 0;
    std::uint32_t serial = 0;
    int ewald_count = 0;
    std::array<int, D> basis{};
};

std::vector<int> representatives_in_mask(const Mask& mask,
                                         const std::vector<std::uint8_t>& is_representative) {
    std::vector<int> result;
    result.reserve(popcount(mask) / 2);
    for (int w = 0; w < WORDS; ++w) {
        std::uint64_t bits = mask[w];
        while (bits) {
            const int offset = __builtin_ctzll(bits);
            const int id = (w << 6) + offset;
            if (id < NX && is_representative[id]) result.push_back(id);
            bits &= bits - 1;
        }
    }
    return result;
}

std::vector<int> all_points_in_mask(const Mask& mask) {
    std::vector<int> result;
    result.reserve(popcount(mask));
    for (int w = 0; w < WORDS; ++w) {
        std::uint64_t bits = mask[w];
        while (bits) {
            const int offset = __builtin_ctzll(bits);
            const int id = (w << 6) + offset;
            if (id < NX) result.push_back(id);
            bits &= bits - 1;
        }
    }
    return result;
}

void write_stored(std::ostream& out, const Stored& s, const std::vector<Point>& points) {
    out << "{\"id\":\"" << poly_id(s.n_facets, s.serial) << "\""
        << ",\"n_facets\":" << s.n_facets
        << ",\"n_ewald_points\":" << s.ewald_count
        << ",\"gf2_rank\":" << s.gf2_rank
        << ",\"basis_found\":" << (s.basis.found ? "true" : "false")
        << ",\"basis_trials\":" << s.basis.trials
        << ",\"sampled_determinant_gcd\":" << s.basis.sampled_gcd
        << ",\"minimum_nonzero_abs_det_seen\":" << s.basis.minimum_nonzero_abs_det
        << ",\"facets\":[";
    for (std::size_t i = 0; i < s.normals.size(); ++i) {
        if (i) out << ',';
        out << normal_json(s.normals[i]);
    }
    out << "]";
    if (s.basis.found) {
        out << ",\"basis\":[";
        for (int i = 0; i < D; ++i) {
            if (i) out << ',';
            out << point_json(points[s.basis.ids[i]]);
        }
        out << ']';
    } else {
        out << ",\"basis\":null";
    }
    out << ",\"ewald_points\":[";
    const std::vector<int> all = all_points_in_mask(s.mask);
    for (std::size_t i = 0; i < all.size(); ++i) {
        if (i) out << ',';
        out << point_json(points[all[i]]);
    }
    out << "]}";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_options(argc, argv);
        char magic[8]{};
        if (!std::cin.read(magic, 8)) throw std::runtime_error("missing input header");
        const char expected_magic[8] = {'E','W','A','L','D','9','\0','\0'};
        if (std::memcmp(magic, expected_magic, 8) != 0) {
            throw std::runtime_error("bad input magic");
        }
        const std::uint64_t expected_records = read_u64(std::cin);

        const std::vector<Point> points = make_points();
        const Mask all = all_mask();
        const int zero_index = (NX - 1) / 2;
        std::vector<std::uint8_t> is_representative(NX, 0);
        std::vector<int> parity(NX, 0);
        for (int i = 0; i < NX; ++i) {
            int first = 0;
            int code = 0;
            for (std::int8_t a : points[i]) {
                if (first == 0 && a != 0) first = a;
                code = (code << 1) | (a != 0 ? 1 : 0);
            }
            is_representative[i] = first > 0 ? 1 : 0;
            parity[i] = code;
        }
        if (!contains(all, zero_index)) throw std::runtime_error("bad zero index");

        RayMaskCache cache(points);
        std::priority_queue<Stored> smallest;
        std::vector<Stored> unresolved;
        std::map<int, std::uint64_t> count_histogram;
        std::map<int, std::uint64_t> facet_histogram;
        std::map<int, std::uint64_t> trial_histogram;
        std::vector<Worst> worst;
        std::uint64_t scanned = 0;
        std::uint64_t bases_found = 0;
        int minimum_ewald = NX;
        int maximum_trials = 0;

        for (; scanned < expected_records; ++scanned) {
            const int n_facets = read_u8(std::cin);
            const std::uint32_t serial = read_u32(std::cin);
            if (n_facets < 10 || n_facets > 26) {
                throw std::runtime_error("invalid facet count in stream");
            }
            std::vector<Normal> normals(static_cast<std::size_t>(n_facets));
            Mask ewald = all;
            for (int i = 0; i < n_facets; ++i) {
                for (int j = 0; j < D; ++j) normals[i][j] = read_i16(std::cin);
                const Mask& constraint = cache.get(normals[i]);
                for (int w = 0; w < WORDS; ++w) ewald[w] &= constraint[w];
            }
            const int n_ewald = popcount(ewald);
            if ((n_ewald & 1) == 0 || !contains(ewald, zero_index)) {
                throw std::runtime_error("invalid Ewald mask at " + poly_id(n_facets, serial));
            }

            const std::vector<int> available = representatives_in_mask(ewald, is_representative);
            const GF2Result gf2 = gf2_basis(available, parity);
            const std::uint64_t seed = 0xd1b54a32d192ed03ULL
                ^ (static_cast<std::uint64_t>(n_facets) << 48)
                ^ (static_cast<std::uint64_t>(serial) * 0x9e3779b97f4a7c15ULL);
            const BasisResult basis = find_basis(gf2, available, seed, opts.max_trials, points);

            ++count_histogram[n_ewald];
            ++facet_histogram[n_facets];
            ++trial_histogram[basis.trials];
            minimum_ewald = std::min(minimum_ewald, n_ewald);
            maximum_trials = std::max(maximum_trials, basis.trials);

            Stored stored;
            stored.n_facets = n_facets;
            stored.serial = serial;
            stored.ewald_count = n_ewald;
            stored.gf2_rank = gf2.rank;
            stored.mask = ewald;
            stored.basis = basis;

            if (basis.found) {
                ++bases_found;
                Worst w;
                w.trials = basis.trials;
                w.n_facets = n_facets;
                w.serial = serial;
                w.ewald_count = n_ewald;
                w.basis = basis.ids;
                worst.push_back(w);
                std::sort(worst.begin(), worst.end(), [](const Worst& a, const Worst& b) {
                    if (a.trials != b.trials) return a.trials > b.trials;
                    if (a.n_facets != b.n_facets) return a.n_facets < b.n_facets;
                    return a.serial < b.serial;
                });
                if (worst.size() > 50) worst.resize(50);
            } else {
                stored.normals = normals;
                unresolved.push_back(stored);
            }

            if (static_cast<int>(smallest.size()) < opts.top_k
                || n_ewald < smallest.top().ewald_count) {
                stored.normals = normals;
                smallest.push(std::move(stored));
                if (static_cast<int>(smallest.size()) > opts.top_k) smallest.pop();
            }

            if ((scanned + 1) % 100000 == 0) {
                std::cerr << opts.shard << ": backend scanned=" << (scanned + 1)
                          << '/' << expected_records
                          << " min=" << minimum_ewald
                          << " unresolved=" << unresolved.size()
                          << " cached_normals=" << cache.size() << '\n';
            }
        }

        // Reject trailing bytes, which would indicate a producer/consumer count mismatch.
        char extra;
        if (std::cin.read(&extra, 1)) throw std::runtime_error("trailing bytes after expected records");

        std::vector<Stored> retained;
        while (!smallest.empty()) {
            retained.push_back(smallest.top());
            smallest.pop();
        }
        std::sort(retained.begin(), retained.end(), [](const Stored& a, const Stored& b) {
            if (a.ewald_count != b.ewald_count) return a.ewald_count < b.ewald_count;
            if (a.n_facets != b.n_facets) return a.n_facets < b.n_facets;
            return a.serial < b.serial;
        });

        std::ofstream out(opts.output);
        if (!out) throw std::runtime_error("cannot open output file");
        out << "{\n"
            << "  \"dimension\":9,\n"
            << "  \"shard\":\"" << opts.shard << "\",\n"
            << "  \"ranges\":\"" << opts.ranges << "\",\n"
            << "  \"complete\":true,\n"
            << "  \"expected_records\":" << expected_records << ",\n"
            << "  \"polytopes_scanned\":" << scanned << ",\n"
            << "  \"bases_found\":" << bases_found << ",\n"
            << "  \"unresolved_count\":" << unresolved.size() << ",\n"
            << "  \"minimum_ewald_points\":" << minimum_ewald << ",\n"
            << "  \"distinct_normal_constraints\":" << cache.size() << ",\n"
            << "  \"maximum_trials\":" << maximum_trials << ",\n"
            << "  \"max_trials_per_polytope\":" << opts.max_trials << ",\n";

        out << "  \"ewald_point_count_histogram\":{";
        bool first = true;
        for (const auto& [value, count] : count_histogram) {
            if (!first) out << ',';
            first = false;
            out << "\n    \"" << value << "\":" << count;
        }
        if (!count_histogram.empty()) out << '\n';
        out << "  },\n";

        out << "  \"facet_count_histogram\":{";
        first = true;
        for (const auto& [value, count] : facet_histogram) {
            if (!first) out << ',';
            first = false;
            out << "\n    \"" << value << "\":" << count;
        }
        if (!facet_histogram.empty()) out << '\n';
        out << "  },\n";

        out << "  \"basis_trial_histogram\":{";
        first = true;
        for (const auto& [value, count] : trial_histogram) {
            if (!first) out << ',';
            first = false;
            out << "\n    \"" << value << "\":" << count;
        }
        if (!trial_histogram.empty()) out << '\n';
        out << "  },\n";

        out << "  \"worst_resolved\":[";
        for (std::size_t i = 0; i < worst.size(); ++i) {
            if (i) out << ',';
            const Worst& w = worst[i];
            out << "\n    {\"id\":\"" << poly_id(w.n_facets, w.serial) << "\""
                << ",\"n_ewald_points\":" << w.ewald_count
                << ",\"trials\":" << w.trials
                << ",\"basis\":[";
            for (int j = 0; j < D; ++j) {
                if (j) out << ',';
                out << point_json(points[w.basis[j]]);
            }
            out << "]}";
        }
        if (!worst.empty()) out << '\n';
        out << "  ],\n";

        out << "  \"smallest_ewald_sets\":[";
        for (std::size_t i = 0; i < retained.size(); ++i) {
            if (i) out << ',';
            out << "\n    ";
            write_stored(out, retained[i], points);
        }
        if (!retained.empty()) out << '\n';
        out << "  ],\n";

        out << "  \"unresolved\":[";
        for (std::size_t i = 0; i < unresolved.size(); ++i) {
            if (i) out << ',';
            out << "\n    ";
            write_stored(out, unresolved[i], points);
        }
        if (!unresolved.empty()) out << '\n';
        out << "  ]\n}\n";

        std::cerr << opts.shard << ": backend complete scanned=" << scanned
                  << " bases=" << bases_found
                  << " unresolved=" << unresolved.size()
                  << " min=" << minimum_ewald
                  << " max_trials=" << maximum_trials << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << '\n';
        return 2;
    }
}
