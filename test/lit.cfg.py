import os
import shutil
import sys
import lit.formats
from lit.llvm import llvm_config
config.name = "Ragged"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.excludes = ["gpu", "Inputs"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.ragged_binary_root, "test")
llvm_config.with_system_environment(["HOME", "TMP", "TEMP", "CUDA_HOME", "CUDA_PATH"])
llvm_config.with_environment("PATH", config.llvm_test_tools_dir, append_path=True)
config.substitutions.append(("%python", sys.executable))
llvm_config.add_tool_substitutions(["ragged-opt", "ragged-compile", "FileCheck", "not"],
    [os.path.join(config.ragged_binary_root, "bin"), config.llvm_test_tools_dir, config.llvm_tools_dir])
if shutil.which("ptxas"):
    config.available_features.add("cuda-ptxas")
