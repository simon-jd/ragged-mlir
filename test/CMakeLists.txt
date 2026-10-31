set(RAGGED_LLVM_TEST_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}" CACHE PATH "Directory containing FileCheck and not")
configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
add_lit_testsuite(check-ragged "Running Ragged MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ragged-opt ragged-compile)
