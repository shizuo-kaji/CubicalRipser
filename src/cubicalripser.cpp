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
#include <queue>
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

#include "cube.h"
#include "dense_cubical_grids.h"
#include "write_pairs.h"
#include "joint_pairs.h"
#include "compute_pairs.h"
#include "config.h"
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
              << "  --algorithm, -a     algorithm to compute the 0-dim persistent homology:\n"
              << "                    link_find      (default)\n"
              << "                    compute_pairs  (slow in most cases)\n"
              << "  --min_recursion_to_cache, -mc  minimum number of recursion for a reduced column to be cached\n"
              << "  --cache_size, -c    maximum number of reduced columns to be cached\n"
              << "  --output, -o        name of the output file\n"
              << "  --print, -p         print persistence pairs on console\n"
              << "  --top_dim          compute only for top dimension using Alexander duality\n"
              << "  --embedded, -e      Take the Alexander dual\n"
              << "  --location, -l      whether creator/destroyer location is included in the output:\n"
              << "                    yes     (default)\n"
              << "                    none\n"
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
                    config_.method = COMPUTEPAIRS;
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

void write_output(const std::vector<WritePairs>& writepairs,
                 const DenseCubicalGrids* dcg,
                 const Config& config) {
    const uint32_t pad_x = (dcg->ax - dcg->img_x) / 2;
    const uint32_t pad_y = (dcg->ay - dcg->img_y) / 2;
    const uint32_t pad_z = (dcg->az - dcg->img_z) / 2;
    const uint32_t pad_w = (dcg->dim < 4) ? 0u : (dcg->aw - dcg->img_w) / 2;

    const auto num_pairs = writepairs.size();
    std::cout << "Total number of pairs: " << num_pairs << std::endl;

    const std::string ext = get_file_extension(config.output_filename);
    if (ext == ".csv") {
        std::ofstream out(config.output_filename.c_str());
        if (!out) {
            throw std::runtime_error("Failed to open output file");
        }

        for (const auto& pair : writepairs) {
            out << static_cast<unsigned int>(pair.dim) << "," << pair.birth << "," << pair.death;
            if (config.location != LOC_NONE) {
                if (dcg->dim < 4) {
                    out << "," << pair.birth_x - pad_x
                        << "," << pair.birth_y - pad_y
                        << "," << pair.birth_z - pad_z
                        << "," << pair.death_x - pad_x
                        << "," << pair.death_y - pad_y
                        << "," << pair.death_z - pad_z;
                } else {
                    out << "," << pair.birth_x - pad_x
                        << "," << pair.birth_y - pad_y
                        << "," << pair.birth_z - pad_z
                        << "," << pair.birth_w - pad_w
                        << "," << pair.death_x - pad_x
                        << "," << pair.death_y - pad_y
                        << "," << pair.death_z - pad_z
                        << "," << pair.death_w - pad_w;
                }
            }
            out << '\n';
        }
    }
    else if (ext == ".npy") {
        const size_t ncols = (dcg->dim < 4) ? 9 : 11; // dim,birth,death,(x1,y1,z1[,w1]),(x2,y2,z2[,w2])
        const std::array<long unsigned, 2> shape = {num_pairs, static_cast<long unsigned>(ncols)};
        std::vector<double> data(ncols * num_pairs, 0.0);

        for (size_t i = 0; i < num_pairs; ++i) {
            const auto& pair = writepairs[i];
            const size_t base = ncols * i;
            data[base + 0] = static_cast<double>(pair.dim);
            data[base + 1] = pair.birth;
            data[base + 2] = pair.death;
            // birth coords
            data[base + 3] = static_cast<double>(pair.birth_x) - pad_x;
            data[base + 4] = static_cast<double>(pair.birth_y) - pad_y;
            data[base + 5] = static_cast<double>(pair.birth_z) - pad_z;
            size_t idx = 6;
            if (dcg->dim >= 4) {
                data[base + idx++] = static_cast<double>(pair.birth_w) - pad_w; // base+6
            }
            // death coords
            data[base + idx++] = static_cast<double>(pair.death_x) - pad_x;
            data[base + idx++] = static_cast<double>(pair.death_y) - pad_y;
            data[base + idx++] = static_cast<double>(pair.death_z) - pad_z;
            if (dcg->dim >= 4) {
                data[base + idx++] = static_cast<double>(pair.death_w) - pad_w; // base+10
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

        std::vector<WritePairs> writepairs;
        std::vector<uint64_t> betti;
        std::vector<Cube> ctr;

        DenseCubicalGrids dcg(config);
        dcg.loadImage(config.embedded);
        config.maxdim = std::min<uint8_t>(config.maxdim, dcg.dim - 1);

        // Compute persistent homology
        switch (config.method) {
            case LINKFIND: {
                Timer timer;
                JointPairs jp(&dcg, writepairs, config);
                // Enumerate edges based on dimension
                if (dcg.dim == 1) {
                    jp.enum_edges({0, 1}, ctr);
                }
                else if (dcg.dim == 2) {
                    jp.enum_edges({0, 1}, ctr);
                }
                else if (dcg.dim == 3) { // 3D
                    jp.enum_edges({0, 1, 2}, ctr);
                }
                else { // 4D
                    jp.enum_edges({0, 1, 2, 3}, ctr);
                }
                // Compute dimension 0 via union-find
                jp.joint_pairs_main(ctr, 0);
                const auto msec = timer.milliseconds();

                betti.push_back(writepairs.size());
                std::cout << "Number of pairs in dim 0: " << betti[0] << std::endl;
                if (config.verbose) {
                    std::cout << "Computation took " << msec << " [msec]" << std::endl;
                }

                // Compute higher dimensions
                if (config.maxdim > 0) {
                    Timer timer1;
                    ComputePairs cp(&dcg, writepairs, config);
                    cp.compute_pairs_main(ctr);  // dim1

                    betti.push_back(writepairs.size() - betti[0]);
                    const auto msec1 = timer1.milliseconds();
                    std::cout << "Number of pairs in dim 1: " << betti[1] << std::endl;
                    if (config.verbose) {
                        std::cout << "Computation took " << msec1 << " [msec]" << std::endl;
                    }

                    if (config.maxdim > 1) {
                        Timer timer2;
                        cp.assemble_columns_to_reduce(ctr, 2);
                        cp.compute_pairs_main(ctr);  // dim2

                        const auto msec2 = timer2.milliseconds();
                        betti.push_back(writepairs.size() - betti[0] - betti[1]);
                        std::cout << "Number of pairs in dim 2: " << betti[2] << std::endl;
                        if (config.verbose) {
                            std::cout << "Computation took " << msec2 << " [msec]" << std::endl;
                        }
                        if (config.maxdim > 2) {
                            Timer timer3;
                            cp.assemble_columns_to_reduce(ctr, 3);
                            cp.compute_pairs_main(ctr);  // dim3

                            const auto msec3 = timer3.milliseconds();
                            betti.push_back(writepairs.size() - betti[0] - betti[1] - betti[2]);
                            std::cout << "Number of pairs in dim 3: " << betti[3] << std::endl;
                            if (config.verbose) {
                                std::cout << "Computation took " << msec3 << " [msec]" << std::endl;
                            }
                        }
                    }
                }

                const auto total_msec = timer.milliseconds();
                std::cout << "Total computation took " << total_msec << " [msec]" << std::endl;
                break;
            }

            case COMPUTEPAIRS: {
                // TODO: bug in T-construction in PH0
                ComputePairs cp(&dcg, writepairs, config);
                // Dimension 0
                cp.assemble_columns_to_reduce(ctr, 0);
                cp.compute_pairs_main(ctr);
                betti.push_back(writepairs.size());
                std::cout << "Number of pairs in dim 0: " << betti[0] << std::endl;

                if (config.maxdim > 0) {
                    // Dimension 1
                    cp.assemble_columns_to_reduce(ctr, 1);
                    cp.compute_pairs_main(ctr);
                    betti.push_back(writepairs.size() - betti[0]);
                    std::cout << "Number of pairs in dim 1: " << betti[1] << std::endl;

                    if (config.maxdim > 1) {
                        // Dimension 2
                        cp.assemble_columns_to_reduce(ctr, 2);
                        cp.compute_pairs_main(ctr);
                        betti.push_back(writepairs.size() - betti[0] - betti[1]);
                        std::cout << "Number of pairs in dim 2: " << betti[2] << std::endl;

                        if (config.maxdim > 2) {
                            // Dimension 3
                            cp.assemble_columns_to_reduce(ctr, 3);
                            cp.compute_pairs_main(ctr);
                            betti.push_back(writepairs.size() - betti[0] - betti[1] - betti[2]);
                            std::cout << "Number of pairs in dim 3: " << betti[3] << std::endl;
                        }
                    }
                }
                break;
            }

            case ALEXANDER: {
                if (config.tconstruction) {
                    throw std::runtime_error("Alexander duality for T-construction not implemented");
                }
                Timer timer;
                JointPairs jp(&dcg, writepairs, config);

                if (dcg.dim == 1) {
                    jp.enum_edges({0}, ctr);
                    jp.joint_pairs_main(ctr, 0);
                    std::cout << "Number of pairs in dim 0: " << writepairs.size() << std::endl;
                }
                else if (dcg.dim == 2) {
                    jp.enum_edges({0, 1, 3, 4}, ctr);
                    jp.joint_pairs_main(ctr, 1);
                    std::cout << "Number of pairs in dim 1: " << writepairs.size() << std::endl;
                }
                else if (dcg.dim == 3) {
                    jp.enum_edges({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, ctr);
                    jp.joint_pairs_main(ctr, 2);
                    std::cout << "Number of pairs in dim 2: " << writepairs.size() << std::endl;
                }
                else if (dcg.dim == 4) {
                    throw std::runtime_error("Alexander duality not implemented for 4D");
                }

                const auto msec = timer.milliseconds();
                std::cout << "Computation took " << msec << " [msec]" << std::endl;
                break;
            }
        }

        write_output(writepairs, &dcg, config);
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
