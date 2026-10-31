add_llvm_executable(ragged-compile ragged-compile.cpp)
llvm_update_compile_flags(ragged-compile)
target_link_libraries(ragged-compile PRIVATE RaggedIR RaggedTransforms
  MLIRParser MLIRPass MLIRRegisterAllDialects MLIRRegisterAllPasses
  MLIRRegisterAllExtensions MLIRToLLVMIRTranslationRegistration)
install(TARGETS ragged-compile RUNTIME DESTINATION bin)
