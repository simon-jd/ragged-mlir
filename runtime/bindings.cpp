#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <tuple>
#include "Driver.h"

namespace nb = nanobind;

NB_MODULE(_native, module) {
  module.doc() = "Ragged CUDA Driver API loader (no CUDA runtime or CPU emulation)";
  nb::class_<ragged::DriverModule>(module, "DriverModule")
      .def("__init__", [](ragged::DriverModule *self, nb::bytes cubin) {
        new (self) ragged::DriverModule(std::string(cubin.c_str(), cubin.size()));
      }, nb::arg("cubin"))
      .def("launch", &ragged::DriverModule::launch,
           nb::arg("kernel"), nb::arg("grid"), nb::arg("block"),
           nb::arg("shared_bytes"), nb::arg("stream"), nb::arg("arguments"))
      .def("resident_blocks", &ragged::DriverModule::residentBlocks,
           nb::arg("kernel"), nb::arg("block_threads"), nb::arg("shared_bytes"))
      .def("add_plan", [](ragged::DriverModule &self,
                          const std::vector<std::tuple<std::string, std::array<unsigned, 3>,
                                                       std::array<unsigned, 3>, unsigned, unsigned>> &steps,
                          const std::vector<std::uint64_t> &scratch,
                          const std::vector<std::uint32_t> &relocations) {
        std::vector<ragged::PlanStep> plan;
        for (const auto &[kernel, grid, block, shared, lane] : steps)
          plan.push_back({kernel, grid, block, shared, lane});
        return self.addPlan(plan, scratch, relocations);
      }, nb::arg("steps"), nb::arg("scratch"), nb::arg("relocations"))
      .def("run_plan", &ragged::DriverModule::runPlan,
           nb::arg("plan"), nb::arg("stream"), nb::arg("inputs"), nb::arg("scratch_base"))
      .def("clear_plans", &ragged::DriverModule::clearPlans)
      .def_prop_ro("device", &ragged::DriverModule::device)
      .def("close", &ragged::DriverModule::close);
}
