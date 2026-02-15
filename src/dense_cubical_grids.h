/* dense_cubical_grids.h

This file is part of CubicalRipser
Copyright 2017-2018 Takeki Sudo and Kazushi Ahara.
Modified by Shizuo Kaji

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <memory>
#include <array>
#include <cstddef>

#include "config.h"
#include "cube.h"
#include "npy.hpp"

using namespace std;

template<typename T>
class NDArray {
private:
    std::vector<T> data_;
    std::vector<size_t> dimensions_;
    std::vector<size_t> strides_;

public:
    NDArray(std::initializer_list<size_t> dims) : dimensions_(dims) {
        size_t total_size = 1;
        strides_.resize(dims.size());

        // Calculate strides (row-major order)
        for (int i = dims.size() - 1; i >= 0; --i) {
            strides_[i] = total_size;
            total_size *= dimensions_[i];
        }
        data_.resize(total_size);
    }

    // Fast paths used by hot loops in getBirth().
    T& operator()(size_t i, size_t j, size_t k) {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2]];
    }

    const T& operator()(size_t i, size_t j, size_t k) const {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2]];
    }

    T& operator()(size_t i, size_t j, size_t k, size_t l) {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]];
    }

    const T& operator()(size_t i, size_t j, size_t k, size_t l) const {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]];
    }

    template<typename... Indices>
    T& operator()(Indices... indices) {
        std::array<size_t, sizeof...(indices)> idx_array = {static_cast<size_t>(indices)...};
        size_t flat_index = 0;
        for (size_t i = 0; i < idx_array.size(); ++i) {
            flat_index += idx_array[i] * strides_[i];
        }
        return data_[flat_index];
    }

    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        std::array<size_t, sizeof...(indices)> idx_array = {static_cast<size_t>(indices)...};
        size_t flat_index = 0;
        for (size_t i = 0; i < idx_array.size(); ++i) {
            flat_index += idx_array[i] * strides_[i];
        }
        return data_[flat_index];
    }
};

class DenseCubicalGrids{
public:
	Config *config;
	double threshold;
	uint8_t dim;
	uint32_t img_x, img_y, img_z, img_w;
	uint32_t ax, ay, az, aw;
    uint32_t axy, axyz, ayz, azw, axw, ayw;
	std::unique_ptr<NDArray<double>> dense;

    DenseCubicalGrids(Config&);
    // Overloaded constructor allowing explicit shape initialization
    DenseCubicalGrids(Config&, uint8_t dim, uint32_t ax, uint32_t ay = 1, uint32_t az = 1, uint32_t aw = 1);
	~DenseCubicalGrids() = default; // NDArray uses RAII, no manual cleanup needed
	double getBirth(uint32_t x, uint32_t y, uint32_t z);
	double getBirth(uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint8_t cm, uint8_t dim);
	vector<uint32_t> ParentVoxel(uint8_t _dim, Cube &c);

	void finalisePadding(){
		// T-construction (the number of vertices = that of the top cells plus one, in each dimension)
		if(config->tconstruction){
			if(dim>3) aw++;
			if(dim>2) az++;
			ax++;
			ay++;
		}
		axy = ax * ay;
		ayz = ay * az;
		azw = az * aw;
		axw = ax * aw;
		ayw = ay * aw;
		axyz = ax * ay * az;
	}

	// load image array from file
	void loadImage(bool embedded){
		// read file
		cout << "Reading " << config->filename << endl;
		switch(config->format){
			case DIPHA:
			{
				ifstream fin( config->filename, ios::in | ios::binary );
				int64_t d;
				fin.read( ( char * ) &d, sizeof( int64_t ) ); // magic number
				assert(d == 8067171840);
				fin.read( ( char * ) &d, sizeof( int64_t ) ); // type number
				assert(d == 1);
				fin.read( ( char * ) &d, sizeof( int64_t ) ); //data num
				fin.read( ( char * ) &d, sizeof( int64_t ) ); // dim
				dim = d;
				assert(dim < 5);
				fin.read( ( char * ) &d, sizeof( int64_t ) );
				ax = d;
				if (dim>1) {
					fin.read( ( char * ) &d, sizeof( int64_t ) );
					ay = d;
				}else{
					ay = 1;
				}
				if (dim>2) {
					fin.read((char *)&d, sizeof(int64_t));
					az = d;
				}else {
					az = 1;
				}
				if (dim>3) {
					fin.read((char *)&d, sizeof(int64_t));
					aw = d;
				}else {
					aw = 1;
				}
				double dou;
				vector<double> arr;
				arr.reserve(ax*ay*az*aw);
				while (!fin.eof()){
					fin.read((char *)&dou, sizeof(double));
					arr.push_back(dou);
				}
				fin.close();
				gridFromArray(&arr[0], embedded, true);
				break;
			}

			case PERSEUS:
			{
				ifstream reading_file;
				reading_file.open(config->filename.c_str(), ios::in);
				string reading_line_buffer;
				getline(reading_file, reading_line_buffer);
				dim = atoi(reading_line_buffer.c_str());
				assert(dim < 5);
				getline(reading_file, reading_line_buffer);
				ax = atoi(reading_line_buffer.c_str());
				if (dim>1) {
					getline(reading_file, reading_line_buffer);
					ay = atoi(reading_line_buffer.c_str());
				}else {
					ay = 1;
				}
				if (dim>2) {
					getline(reading_file, reading_line_buffer);
					az = atoi(reading_line_buffer.c_str());
				}else {
					az = 1;
				}
				if (dim>3) {
					getline(reading_file, reading_line_buffer);
					aw = atoi(reading_line_buffer.c_str());
				}else {
					aw = 1;
				}
				vector<double> arr;
				arr.reserve(ax*ay*az*aw);
				while(!reading_file.eof()){
					getline(reading_file, reading_line_buffer);
					double dou = atof(reading_line_buffer.c_str());
					if (dou != -1) {
						arr.push_back(dou);
					}else{
						arr.push_back(config->threshold);
					}
				}
				reading_file.close();
				gridFromArray(&arr[0], embedded, true);
				break;
			}

			case CSV:
			{
				dim = 2;
				vector<double> arr;
				ifstream reading_file;
				reading_file.open(config->filename.c_str(), ios::in);
				string line;
				ay = 0;
				while (getline(reading_file, line)) {
					istringstream stream(line);
					string field;
					ax = 0;
					while (getline(stream, field, ',')) {
						arr.push_back(stod(field));
						ax++;
					}
					ay++;
				}
				az = 1;
				aw = 1;
				gridFromArray(&arr[0], embedded, true);
				break;
			}

			case NUMPY:
			{
				vector<unsigned long> shape;
				vector<double> arr;
				bool fortran_order;
				try{
					npy::LoadArrayFromNumpy(config->filename.c_str(), shape, fortran_order, arr);
				} catch (...) {
					cerr << "The data type of an numpy array should be numpy.float64." << endl;
					exit(-2);
				}
				if(shape.size() > 4){
					cerr << "Input array should be 1,2,3, or 4 dimensional " << endl;
					exit(-1);
				}
				dim = shape.size();
				ax = shape[0];
				if (dim>1) {
					ay = shape[1];
				}else {
					ay = 1;
				}
				if (dim>2) {
					az = shape[2];
				}else {
					az = 1;
				}
				if (dim>3) {
					aw = shape[3];
				}else {
					aw = 1;
				}
				gridFromArray(&arr[0], embedded, fortran_order);
				break;
			}
		}
		cout << "dim = " << static_cast<int>(dim) << " T-construction = " << config->tconstruction << " method = " << config->method << endl;
		finalisePadding();
		if (dim < 4){
			cout << "x : y : z = " << img_x << " : " << img_y << " : " << img_z << endl;
			//cout << "x : y : z = " << ax << " : " << ay << " : " << az << endl;
		}else{
			cout << "x : y : z : w = " << img_x << " : " << img_y << " : " << img_z << " : " << img_w << endl;
			//cout << "x : y : z : w = " << ax << " : " << ay << " : " << az << " : " << aw << endl;
		}
	}

	// construct volume with boundary
	void gridFromArray(const double *arr, bool embedded, bool fortran_order){
		img_x = ax;
		img_y = ay;
		img_z = az;
		img_w = aw;
		uint64_t i = 0;
		uint32_t x_shift = 2; // total size of the boundary (left+right)
		uint32_t y_shift = 2;
		uint32_t z_shift = 2;
		uint32_t w_shift = 2; // 4D case
		double sgn = 1;
		if(embedded){
			sgn = -1;
			x_shift = 4; // 2 inner + 2 outer
			y_shift = 4;
			if (az>1) z_shift = 4;
			if (aw>1) w_shift = 4;
		}
		if (dim < 4){
			// allocate with inner&outer boundary (original ax,ay,az not yet modified below)
			const uint32_t size_x = ax + x_shift;
			const uint32_t size_y = ay + y_shift;
			const uint32_t size_z = az + z_shift;
			dense = std::make_unique<NDArray<double>>(std::initializer_list<size_t>{size_x, size_y, size_z});

			const uint32_t inner_x_begin = x_shift / 2;
			const uint32_t inner_y_begin = y_shift / 2;
			const uint32_t inner_z_begin = z_shift / 2;
			const uint32_t inner_x_end_ex = inner_x_begin + ax; // exclusive
			const uint32_t inner_y_end_ex = inner_y_begin + ay;
			const uint32_t inner_z_end_ex = inner_z_begin + az;

			auto arrIndexFortran = [&](uint32_t ox, uint32_t oy, uint32_t oz) -> size_t {
				// Fortran order: first index (x) fastest
				return static_cast<size_t>(ox) + static_cast<size_t>(ax) * ( static_cast<size_t>(oy) + static_cast<size_t>(ay) * static_cast<size_t>(oz) );
			};
			auto arrIndexC = [&](uint32_t ox, uint32_t oy, uint32_t oz) -> size_t {
				// C order: last index (z) fastest
				return static_cast<size_t>(oz) + static_cast<size_t>(az) * ( static_cast<size_t>(oy) + static_cast<size_t>(ay) * static_cast<size_t>(ox) );
			};

			for (uint32_t x = 0; x < size_x; ++x){
				bool inner_x = (x >= inner_x_begin && x < inner_x_end_ex);
				uint32_t ox = x - inner_x_begin;
				for (uint32_t y = 0; y < size_y; ++y){
					bool inner_y = (y >= inner_y_begin && y < inner_y_end_ex);
					uint32_t oy = y - inner_y_begin;
					for (uint32_t z = 0; z < size_z; ++z){
						bool inner_z = (z >= inner_z_begin && z < inner_z_end_ex);
						uint32_t oz = z - inner_z_begin;

						if (inner_x && inner_y && inner_z){
							size_t idx = fortran_order
								? arrIndexFortran(ox, oy, oz)
								: arrIndexC(ox, oy, oz);
							(*dense)(x, y, z) = sgn * arr[idx];
						}else{
							// outer boundary
							if (x == 0 || x == size_x - 1 ||
								y == 0 || y == size_y - 1 ||
								z == 0 || z == size_z - 1){
								(*dense)(x, y, z) = config->threshold;
							}else{ // inner boundary (only when embedded)
								(*dense)(x, y, z) = -config->threshold;
							}
						}
					}
				}
			}
		} else {
			// 4D case
			const uint32_t size_x = ax + x_shift;
			const uint32_t size_y = ay + y_shift;
			const uint32_t size_z = az + z_shift;
			const uint32_t size_w = aw + w_shift;
			dense = std::make_unique<NDArray<double>>(std::initializer_list<size_t>{size_x, size_y, size_z, size_w});

			const uint32_t inner_x_begin = x_shift / 2;
			const uint32_t inner_y_begin = y_shift / 2;
			const uint32_t inner_z_begin = z_shift / 2;
			const uint32_t inner_w_begin = w_shift / 2;
			const uint32_t inner_x_end_ex = inner_x_begin + ax; // exclusive
			const uint32_t inner_y_end_ex = inner_y_begin + ay;
			const uint32_t inner_z_end_ex = inner_z_begin + az;
			const uint32_t inner_w_end_ex = inner_w_begin + aw;

			auto arrIndexFortran4D = [&](uint32_t ox, uint32_t oy, uint32_t oz, uint32_t ow) -> size_t {
				// Fortran order: first index (x) fastest
				return static_cast<size_t>(ox) + static_cast<size_t>(ax) * (
					static_cast<size_t>(oy) + static_cast<size_t>(ay) * (
						static_cast<size_t>(oz) + static_cast<size_t>(az) * static_cast<size_t>(ow) ));
			};
			auto arrIndexC4D = [&](uint32_t ox, uint32_t oy, uint32_t oz, uint32_t ow) -> size_t {
				// C order: last index (w) fastest
				return static_cast<size_t>(ow) + static_cast<size_t>(aw) * (
					static_cast<size_t>(oz) + static_cast<size_t>(az) * (
						static_cast<size_t>(oy) + static_cast<size_t>(ay) * static_cast<size_t>(ox) ));
			};

			for (uint32_t x = 0; x < size_x; ++x){
				bool inner_x = (x >= inner_x_begin && x < inner_x_end_ex);
				uint32_t ox = x - inner_x_begin;
				for (uint32_t y = 0; y < size_y; ++y){
					bool inner_y = (y >= inner_y_begin && y < inner_y_end_ex);
					uint32_t oy = y - inner_y_begin;
					for (uint32_t z = 0; z < size_z; ++z){
						bool inner_z = (z >= inner_z_begin && z < inner_z_end_ex);
						uint32_t oz = z - inner_z_begin;
						for (uint32_t w = 0; w < size_w; ++w){
							bool inner_w = (w >= inner_w_begin && w < inner_w_end_ex);
							uint32_t ow = w - inner_w_begin;

							if (inner_x && inner_y && inner_z && inner_w){
								size_t idx = fortran_order
									? arrIndexFortran4D(ox, oy, oz, ow)
									: arrIndexC4D(ox, oy, oz, ow);
								(*dense)(x, y, z, w) = sgn * arr[idx];
							}else{
								// outer boundary
								if (x == 0 || x == size_x - 1 ||
									y == 0 || y == size_y - 1 ||
									z == 0 || z == size_z - 1 ||
									w == 0 || w == size_w - 1){
									(*dense)(x, y, z, w) = config->threshold;
								}else{ // inner boundary (only when embedded)
									(*dense)(x, y, z, w) = -config->threshold;
								}
							}
						}
					}
				}
			}
		}
		ax += x_shift-2;
		ay += y_shift-2;
		az += z_shift-2;
		aw += w_shift-2;
	}
};
