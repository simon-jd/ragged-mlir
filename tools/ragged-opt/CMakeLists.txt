add_llvm_executable(ragged-opt ragged-opt.cpp)
llvm_update_compile_flags(ragged-opt)
target_link_libraries(ragged-opt PRIVATE RaggedIR RaggedTransforms
  MLIROptLib MLIRRegisterAllDialects MLIRRegisterAllPasses MLIRRegisterAllExtensions
  MLIRToLLVMIRTranslationRegistration)
install(TARGETS ragged-opt RUNTIME DESTINATION bin)
