/*
This file is part of CubicalRipser
Copyright 2017-2018 Takeki Sudo and Kazushi Ahara.
Modified by Shizuo Kaji

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along
with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <memory>
#include <sstream>
#include <array>
#include <iomanip>
#include <limits>
#include <map>

#include "cube.h"
#include "dense_cubical_grids.h"
#include "write_pairs.h"
#include "config.h"
#include "persistence.h"
#include "image_io.h"
#include "npy.hpp"

namespace {

class Timer {
    using Clock = std::chrono::system_clock;
    using TimePoint = Clock::time_point;

public:
    Timer() : start_(Clock::now()) {}

    [[nodiscard]] int64_t milliseconds() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - start_
        ).count();
    }

private:
    TimePoint start_;
};

void print_usage() {
    std::cerr << "Usage: cubicalripser [options] [input_filename]\n"
              << "\nOptions:\n\n"
              << "  --help, -h          print this screen\n"
              << "  --verbose, -v       enable verbose output\n"
              << "  --threshold <t>, -t compute cubical complexes up to birth time <t>\n"
              << "  --maxdim <t>, -m    compute persistent homology up to dimension <t>\n"
              << "  --threads <n>       worker threads for grid scans and sorts\n"
              << "                      (1 = sequential, default; 0 = auto). Output is identical either way.\n"
              << "  --algorithm, -a     algorithm for the 0-dim persistent homology; only\n"
              << "                    link_find (union-find) is available\n"
              << "  --min_recursion_to_cache, -mc  minimum number of recursion for a reduced column to be cached\n"
              << "  --cache_size, -c    maximum number of reduced columns to be cached\n"
              << "  --output, -o        name of the output file\n"
              << "  --print, -p         print persistence pairs on console\n"
              << "  --top_dim          compute only for top dimension using Alexander duality\n"
              << "  --embedded, -e      Take the Alexander dual\n"
              << "  --location, -l      whether creator/destroyer location is included in the output:\n"
              << "                    yes     (default)\n"
              << "                    none\n"
              << "  --vector-working-column  use sorted-vector working columns for 4D H1 (experimental)\n"
              << "  --explicit-clearing compress pivot table to a bitset before next-dim clearing (experimental; default)\n"
              << "  --no-explicit-clearing disable explicit clearing\n"
              << std::endl;
}

class ArgumentParser {
public:
    explicit ArgumentParser(int argc, char** argv) {
        parse(argc, argv);
    }

    const Config& get_config() const { return config_; }

private:
    Config config_;

    void parse(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);

            if (arg == "--help" || arg == "-h") {
                print_usage();
                std::exit(0);
            }
            else if (arg == "--verbose" || arg == "-v") {
                config_.verbose = true;
            }
            else if (arg == "--threshold" || arg == "-t") {
                if (i + 1 >= argc) throw std::runtime_error("Missing threshold value");
                try {
                    config_.threshold = std::stod(argv[++i]);
                } catch (const std::exception& e) {
                    throw std::runtime_error("Invalid threshold value");
                }
            }
            else if (arg == "--maxdim" || arg == "-m") {
                if (i + 1 >= argc) throw std::runtime_error("Missing maxdim value");
                try {
                    config_.maxdim = std::stoi(argv[++i]);
                } catch (const std::exception& e) {
                    throw std::runtime_error("Invalid maxdim value");
                }
            }
            else if (arg == "--algorithm" || arg == "-a") {
                if (i + 1 >= argc) throw std::runtime_error("Missing algorithm value");
                std::string param(argv[++i]);
                if (param == "link_find") {
                    config_.method = LINKFIND;
                }
                else if (param == "compute_pairs") {
                    throw std::runtime_error(
                        "The compute_pairs algorithm has been removed; use link_find");
                }
                else {
                    throw std::runtime_error("Invalid algorithm value");
                }
            }
            else if (arg == "--output" || arg == "-o") {
                if (i + 1 >= argc) throw std::runtime_error("Missing output filename");
                config_.output_filename = argv[++i];
            }
            else if (arg == "--min_recursion_to_cache" || arg == "-mc") {
                if (i + 1 >= argc) throw std::runtime_error("Missing min recursion value");
                try {
                    config_.min_recursion_to_cache = std::stoi(argv[++i]);
                } catch (const std::exception& e) {
                    throw std::runtime_error("Invalid min recursion value");
                }
            }
            else if (arg == "--cache_size" || arg == "-c") {
                if (i + 1 >= argc) throw std::runtime_error("Missing cache size value");
                try {
                    config_.cache_size = std::stoi(argv[++i]);
                } catch (const std::exception& e) {
                    throw std::runtime_error("Invalid cache size value");
                }
            }
            else if (arg == "--print" || arg == "-p") {
                config_.print = true;
            }
            else if (arg == "--embedded" || arg == "-e") {
                config_.embedded = true;
            }
            else if (arg == "--top_dim") {
                config_.method = ALEXANDER;
            }
            else if (arg == "--threads") {
                if (i + 1 >= argc) throw std::runtime_error("Missing threads value");
                try {
                    config_.num_threads = std::stoi(argv[++i]);
                } catch (const std::exception&) {
                    throw std::runtime_error("Invalid threads value");
                }
            }
            else if (arg == "--vector-working-column") {
                config_.vector_working_column = true;
            }
            else if (arg == "--explicit-clearing") {
                config_.explicit_clearing = true;
            }
            else if (arg == "--no-explicit-clearing") {
                config_.explicit_clearing = false;
            }
            else if (arg == "--location" || arg == "-l") {
                if (i + 1 >= argc) throw std::runtime_error("Missing location value");
                std::string param(argv[++i]);
                if (param == "none") {
                    config_.location = LOC_NONE;
                }
                else if (param != "yes") {
                    throw std::runtime_error("Invalid location value");
                }
            }
            else {
                if (!arg.empty() && arg[0] == '-') {
                    throw std::runtime_error("Unknown option: " + arg);
                }
                if (!config_.filename.empty()) {
                    throw std::runtime_error("Multiple input files specified");
                }
                config_.filename = argv[i];
            }
        }

        if (config_.filename.empty()) {
            throw std::runtime_error("No input file specified");
        }
    }
};

std::string get_file_extension(const std::string& filename) {
    size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos) return "";
    return filename.substr(pos);
}

void determine_file_format(Config& config) {
    static const std::unordered_map<std::string, file_format> format_map{{".txt", PERSEUS},
                                                                        {".npy", NUMPY},
                                                                        {".csv", CSV},
                                                                        {".complex", DIPHA}};

    std::string ext = get_file_extension(config.filename);
    // Convert to lowercase
    std::transform(ext.begin(), ext.end(), ext.begin(),
                  [](unsigned char c){ return std::tolower(c); });

    auto it = format_map.find(ext);
    if (it == format_map.end()) {
        throw std::runtime_error(
            "Unknown input file format (supported: .npy, .txt, .csv, .complex)");
    }
    config.format = it->second;
}

bool file_exists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

void write_output(const Persistence& persistence, uint8_t dim,
                 const Config& config) {
    const std::vector<WritePairs>& writepairs = persistence.pairs;
    const InputCoordinates& input_voxel = persistence.input_voxel;
    const int num_axes = (dim < 4) ? 3 : 4;

    const auto num_pairs = writepairs.size();
    std::cout << "Total number of pairs: " << num_pairs << std::endl;

    const std::string ext = get_file_extension(config.output_filename);
    if (ext == ".csv") {
        std::ofstream out(config.output_filename.c_str());
        if (!out) {
            throw std::runtime_error("Failed to open output file");
        }
        out << std::setprecision(std::numeric_limits<double>::max_digits10);

        for (const auto& pair : writepairs) {
            out << static_cast<unsigned int>(pair.dim) << "," << pair.birth << "," << pair.death;
            if (config.location != LOC_NONE) {
                const auto b = input_voxel(pair.birth_x, pair.birth_y, pair.birth_z, pair.birth_w);
                const auto d = input_voxel(pair.death_x, pair.death_y, pair.death_z, pair.death_w);
                for (int a = 0; a < num_axes; ++a) out << "," << b[a];
                for (int a = 0; a < num_axes; ++a) out << "," << d[a];
            }
            out << '\n';
        }
    }
    else if (ext == ".npy") {
        const size_t ncols = 3 + 2 * num_axes; // dim,birth,death,(x1,y1,z1[,w1]),(x2,y2,z2[,w2])
        const std::array<long unsigned, 2> shape = {num_pairs, static_cast<long unsigned>(ncols)};
        std::vector<double> data(ncols * num_pairs, 0.0);

        for (size_t i = 0; i < num_pairs; ++i) {
            const auto& pair = writepairs[i];
            const size_t base = ncols * i;
            data[base + 0] = static_cast<double>(pair.dim);
            data[base + 1] = pair.birth;
            data[base + 2] = pair.death;
            const auto b = input_voxel(pair.birth_x, pair.birth_y, pair.birth_z, pair.birth_w);
            const auto d = input_voxel(pair.death_x, pair.death_y, pair.death_z, pair.death_w);
            for (int a = 0; a < num_axes; ++a) {
                data[base + 3 + a] = static_cast<double>(b[a]);
                data[base + 3 + num_axes + a] = static_cast<double>(d[a]);
            }
        }

        try {
            npy::SaveArrayAsNumpy(config.output_filename, false, 2, shape.data(), data);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to write NPY file: " + std::string(e.what()));
        }
    }
    else if (config.output_filename != "none") {
        std::ofstream out(config.output_filename.c_str(), std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open output file");
        }

        const int64_t magic_number = 8067171840;
        const int64_t type = 2;  // PERSISTENCE_DIAGRAM
        const int64_t num_points = num_pairs;

        out.write(reinterpret_cast<const char*>(&magic_number), sizeof(int64_t));
        out.write(reinterpret_cast<const char*>(&type), sizeof(int64_t));
        out.write(reinterpret_cast<const char*>(&num_points), sizeof(int64_t));

        for (const auto& pair : writepairs) {
            const int64_t dim = pair.dim;
            out.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
            out.write(reinterpret_cast<const char*>(&pair.birth), sizeof(double));
            out.write(reinterpret_cast<const char*>(&pair.death), sizeof(double));
        }
    }
}

} // anonymous namespace

int main(int argc, char** argv) {
    try {
        ArgumentParser parser(argc, argv);
        auto& config = const_cast<Config&>(parser.get_config());

        if (!file_exists(config.filename)) {
            throw std::runtime_error("Input file not found: " + config.filename);
        }

        determine_file_format(config);

        std::cout << "Reading " << config.filename << std::endl;
        const InputImage image = read_input_image(config);
        DenseCubicalGrids dcg(config, image.dim, image.shape[0], image.shape[1],
                              image.shape[2], image.shape[3]);
        std::cout << "dim = " << static_cast<int>(dcg.dim)
                  << " T-construction = " << config.tconstruction
                  << " method = " << config.method << std::endl;
        std::cout << "x : y : z" << (dcg.dim < 4 ? "" : " : w") << " = " << image.shape[0]
                  << " : " << image.shape[1] << " : " << image.shape[2];
        if (dcg.dim == 4) std::cout << " : " << image.shape[3];
        std::cout << std::endl;
        config.maxdim = std::min<uint8_t>(config.maxdim, dcg.dim - 1);

        Timer timer;
        const Persistence persistence = compute_persistence(
            dcg, image.values.data(), image.fortran_order, config);
        std::map<unsigned, uint64_t> pairs_per_dim;
        for (const auto& pair : persistence.pairs) ++pairs_per_dim[pair.dim];
        for (const auto& [dim, count] : pairs_per_dim) {
            std::cout << "Number of pairs in dim " << dim << ": " << count << std::endl;
        }
        std::cout << "Total computation took " << timer.milliseconds() << " [msec]" << std::endl;

        write_output(persistence, dcg.dim, config);
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
