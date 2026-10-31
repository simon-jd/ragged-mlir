// RUN: ragged-compile --toolchain-json | %python %S/Inputs/check-provenance.py
// RUN: %python %S/Inputs/check-provenance-cmake.py %S/../..
