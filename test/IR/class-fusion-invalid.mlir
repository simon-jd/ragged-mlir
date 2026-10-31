// RUN: ragged-opt %s --split-input-file --verify-diagnostics
// The per-class softmax schedule must agree with the dispatch classes.

func.func @mixed_without_schedule(%v: memref<?x16xf32>, %o: memref<?xi64>, %out: memref<?x16xf32>) {
  "ragged.dispatch"() ({
    "ragged.classify"(%o) {thresholds = array<i64: 32, 4096>} : (memref<?xi64>) -> ()
    // expected-error @+1 {{mixed fusion requires class_fusion}}
    "ragged.map_segments"(%v, %o, %out) {operation = "softmax", accumulator = f32, identity = "negative_infinity", nan = "propagate", semantics_version = 1 : i64, fusion = "mixed"} : (memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>) -> ()
    "ragged.yield"() : () -> ()
  }) {classes = ["warp", "cta", "split"], thresholds = array<i64: 32, 4096>, threads = 128 : i64, d_tile = 16 : i64, shared_bytes = 1024 : i64} : () -> ()
  return
}

// -----

func.func @summary_disagrees(%v: memref<?x16xf32>, %o: memref<?xi64>, %out: memref<?x16xf32>) {
  "ragged.dispatch"() ({
    "ragged.classify"(%o) {thresholds = array<i64: 32, 4096>} : (memref<?xi64>) -> ()
    // expected-error @+1 {{fusion must summarize class_fusion}}
    "ragged.map_segments"(%v, %o, %out) {operation = "softmax", accumulator = f32, identity = "negative_infinity", nan = "propagate", semantics_version = 1 : i64, fusion = "online", class_fusion = ["online", "online", "three_pass"]} : (memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>) -> ()
    "ragged.yield"() : () -> ()
  }) {classes = ["warp", "cta", "split"], thresholds = array<i64: 32, 4096>, threads = 128 : i64, d_tile = 16 : i64, shared_bytes = 1024 : i64} : () -> ()
  return
}

// -----

func.func @short_schedule(%v: memref<?x16xf32>, %o: memref<?xi64>, %out: memref<?x16xf32>) {
  "ragged.dispatch"() ({
    "ragged.classify"(%o) {thresholds = array<i64: 32, 4096>} : (memref<?xi64>) -> ()
    // expected-error @+1 {{class_fusion must have one entry per dispatch class}}
    "ragged.map_segments"(%v, %o, %out) {operation = "softmax", accumulator = f32, identity = "negative_infinity", nan = "propagate", semantics_version = 1 : i64, fusion = "mixed", class_fusion = ["online", "three_pass"]} : (memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>) -> ()
    "ragged.yield"() : () -> ()
  }) {classes = ["warp", "cta", "split"], thresholds = array<i64: 32, 4096>, threads = 128 : i64, d_tile = 16 : i64, shared_bytes = 1024 : i64} : () -> ()
  return
}

// -----

func.func @unknown_schedule(%v: memref<?x16xf32>, %o: memref<?xi64>, %out: memref<?x16xf32>) {
  "ragged.dispatch"() ({
    "ragged.classify"(%o) {thresholds = array<i64: 32, 4096>} : (memref<?xi64>) -> ()
    // expected-error @+1 {{class_fusion must list 'online' or 'three_pass' per class, with fusion}}
    "ragged.map_segments"(%v, %o, %out) {operation = "softmax", accumulator = f32, identity = "negative_infinity", nan = "propagate", semantics_version = 1 : i64, fusion = "mixed", class_fusion = ["online", "fused", "three_pass"]} : (memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>) -> ()
    "ragged.yield"() : () -> ()
  }) {classes = ["warp", "cta", "split"], thresholds = array<i64: 32, 4096>, threads = 128 : i64, d_tile = 16 : i64, shared_bytes = 1024 : i64} : () -> ()
  return
}

// -----

func.func @schedule_without_summary(%v: memref<?x16xf32>, %o: memref<?xi64>, %out: memref<?x16xf32>) {
  "ragged.dispatch"() ({
    "ragged.classify"(%o) {thresholds = array<i64: 32, 4096>} : (memref<?xi64>) -> ()
    // expected-error @+1 {{class_fusion must list 'online' or 'three_pass' per class, with fusion}}
    "ragged.map_segments"(%v, %o, %out) {operation = "softmax", accumulator = f32, identity = "negative_infinity", nan = "propagate", semantics_version = 1 : i64, class_fusion = ["three_pass", "three_pass", "three_pass"]} : (memref<?x16xf32>, memref<?xi64>, memref<?x16xf32>) -> ()
    "ragged.yield"() : () -> ()
  }) {classes = ["warp", "cta", "split"], thresholds = array<i64: 32, 4096>, threads = 128 : i64, d_tile = 16 : i64, shared_bytes = 1024 : i64} : () -> ()
  return
}
