// RUN: ragged-opt %s --split-input-file --ragged-pipeline --verify-diagnostics

func.func @nonboolean_debug(%v: memref<?x7xf32>, %o: memref<?xi64>, %out: memref<?x7xf32>) {
  // expected-error @+1 {{debug_visits must be a boolean}}
  ragged.segment_reduce %v, %o into %out {kind = "sum", debug_visits = 1 : i64} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
  return
}

// -----

func.func @split_missing_classifier(%v: memref<?x7xf32>, %o: memref<?xi64>, %out: memref<?x7xf32>) {
  // expected-error @+1 {{requires one map and a classifier exactly when device classification is active}}
  "ragged.dispatch"() ({
    ragged.map_segments %v, %o into %out {operation = "sum", accumulator = f32, identity = "zero", nan = "propagate", semantics_version = 1 : i64} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
    ragged.yield
  }) {classes = ["split"], thresholds = array<i64: 32, 4096>, threads = 128 : i64, d_tile = 8 : i64, shared_bytes = 512 : i64} : () -> ()
  return
}

// -----

func.func @debug_warp_missing_classifier(%v: memref<?x7xf32>, %o: memref<?xi64>, %out: memref<?x7xf32>) {
  // expected-error @+1 {{requires one map and a classifier exactly when device classification is active}}
  "ragged.dispatch"() ({
    ragged.map_segments %v, %o into %out {operation = "sum", accumulator = f32, identity = "zero", nan = "propagate", semantics_version = 1 : i64, debug_visits = true} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
    ragged.yield
  }) {classes = ["warp"], thresholds = array<i64: 32, 4096>, threads = 32 : i64, d_tile = 8 : i64, shared_bytes = 128 : i64} : () -> ()
  return
}

// -----

func.func @classification_flag_lies(%v: memref<?x7xf32>, %o: memref<?xi64>, %out: memref<?x7xf32>) {
  // expected-error @+1 {{device_classification must match classes and debug_visits}}
  "ragged.dispatch"() ({
    ragged.classify %o {thresholds = array<i64: 32, 4096>} : memref<?xi64>
    ragged.map_segments %v, %o into %out {operation = "sum", accumulator = f32, identity = "zero", nan = "propagate", semantics_version = 1 : i64} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
    ragged.yield
  }) {classes = ["warp", "cta"], thresholds = array<i64: 32, 4096>, device_classification = false, threads = 128 : i64, d_tile = 8 : i64, shared_bytes = 512 : i64} : () -> ()
  return
}

// -----

// expected-error @+1 {{classified workspace ABI requires exactly one dispatch per module}}
module {
  func.func @two_programs(%v: memref<?x7xf32>, %o: memref<?xi64>, %out: memref<?x7xf32>) {
    ragged.segment_reduce %v, %o into %out {kind = "sum"} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
    ragged.segment_reduce %v, %o into %out {kind = "max"} : memref<?x7xf32>, memref<?xi64>, memref<?x7xf32>
    return
  }
}
