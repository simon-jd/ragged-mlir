// RUN: ragged-opt %s --split-input-file --verify-diagnostics

func.func @wide_hint(%v: memref<?x16xf32>, %o: memref<?xi64>, %out: memref<?x16xf32>) {
  // expected-error @+1 {{max_len must be a nonnegative i64 integer}}
  "ragged.segment_reduce"(%v, %o, %out) {kind = "sum", max_len = 18446744073709551616 : i128} : (memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>) -> ()
  return
}

// -----

func.func @malformed_class(%v: memref<?x16xf32>, %o: memref<?xi64>, %out: memref<?x16xf32>) {
  // expected-error @+1 {{classes must be unique strings ordered warp, cta, split}}
  "ragged.dispatch"() ({
    "ragged.map_segments"(%v, %o, %out) {operation = "sum", accumulator = f32, nan = "propagate"} : (memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>) -> ()
    "ragged.yield"() : () -> ()
  }) {classes = [1 : i64], thresholds = array<i64: 32, 4096>, threads = 128 : i64, d_tile = 16 : i64, shared_bytes = 512 : i64} : () -> ()
  return
}

// -----

func.func @bad_tree_shape(%v: memref<?x16xf32>, %o: memref<?xi64>, %out: memref<?x16xf32>) {
  // expected-error @+1 {{threads must be a power of two}}
  "ragged.dispatch"() ({
    "ragged.map_segments"(%v, %o, %out) {operation = "max", accumulator = f32, nan = "propagate"} : (memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>) -> ()
    "ragged.yield"() : () -> ()
  }) {classes = ["cta"], thresholds = array<i64: 32, 4096>, threads = 96 : i64, d_tile = 16 : i64, shared_bytes = 512 : i64} : () -> ()
  return
}

// -----

func.func @classifier_other_offsets(%v: memref<?x16xf32>, %o: memref<?xi64>, %other: memref<?xi64>, %out: memref<?x16xf32>) {
  // expected-error @+1 {{classify must use the mapped offsets and dispatch thresholds}}
  "ragged.dispatch"() ({
    "ragged.classify"(%other) {thresholds = array<i64: 32, 4096>} : (memref<?xi64>) -> ()
    "ragged.map_segments"(%v, %o, %out) {operation = "sum", accumulator = f32, nan = "propagate"} : (memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>) -> ()
    "ragged.yield"() : () -> ()
  }) {classes = ["warp", "cta"], thresholds = array<i64: 32, 4096>, threads = 128 : i64, d_tile = 16 : i64, shared_bytes = 512 : i64} : () -> ()
  return
}

// -----

func.func @actual_alias_despite_declaration(%v: memref<?x16xf32>, %o: memref<?xi64>) {
  // expected-error @+1 {{in-place reduction is unsafe}}
  "ragged.segment_reduce"(%v, %o, %v) {kind = "sum", alias = "none"} : (memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>) -> ()
  return
}
