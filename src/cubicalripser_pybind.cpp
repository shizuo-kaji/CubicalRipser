/*
This file is part of CubicalRipser
Copyright 2017-2018 Takeki Sudo and Kazushi Ahara.
Modified by Shizuo Kaji

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "cubicalripser_pybind.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;

namespace py = pybind11;

// Allow building two modules from this single source by parameterizing
// the module name and doc metadata via compile definitions.
#ifndef CRIPSER_MODULE_NAME
#  define CRIPSER_MODULE_NAME _cripser
#endif

#ifndef CRIPSER_MODULE_DOC
#  define CRIPSER_MODULE_DOC "Cubical Ripser plugin"
#endif

#ifndef CRIPSER_CURRENTMODULE
#  define CRIPSER_CURRENTMODULE "cripser"
#endif

PYBIND11_MODULE(CRIPSER_MODULE_NAME, m) {
    m.doc() =
        CRIPSER_MODULE_DOC "\n"
        "-----------------------\n"
        ".. currentmodule:: " CRIPSER_CURRENTMODULE "\n"
        ".. autosummary::\n"
        "   :toctree: _generate\n"
        "   add\n"
        "   subtract\n";

    m.def("computePH", &computePH, "Compute Persistent Homology",
          py::arg("arr"),  py::arg("maxdim")=2, py::arg("top_dim")=false,
          py::arg("embedded")=false, py::arg("location")="yes");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
