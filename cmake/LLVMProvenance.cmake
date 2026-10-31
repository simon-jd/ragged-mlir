# Release archives do not carry .git and upstream consequently generates
# VCSRevision.h with LLVM_REVISION undefined. Do not guess a revision from the
# version string: accept the locked release's bytes as a separately labelled
# base identity, or require the SDK's own revision header.
set(RAGGED_LLVM_SOURCE_ARCHIVE "" CACHE FILEPATH
    "Official LLVM source archive used to build an SDK without LLVM_REVISION")
set(RAGGED_VERIFIED_LLVM_REVISION "")
set(RAGGED_LLVM_SOURCE_ARCHIVE_SHA256 "")
if(RAGGED_LLVM_SOURCE_ARCHIVE)
  if(NOT EXISTS "${RAGGED_LLVM_SOURCE_ARCHIVE}")
    message(FATAL_ERROR "RAGGED_LLVM_SOURCE_ARCHIVE does not exist: ${RAGGED_LLVM_SOURCE_ARCHIVE}")
  endif()
  file(SHA256 "${RAGGED_LLVM_SOURCE_ARCHIVE}" RAGGED_LLVM_SOURCE_ARCHIVE_SHA256)
  if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL "23.1.3" OR
     NOT RAGGED_LLVM_SOURCE_ARCHIVE_SHA256 STREQUAL "c44186a7762ed28954be72e5ff6df9808e0779d4f1bf014ecc4e7e211d31ee34")
    message(FATAL_ERROR "Unrecognized LLVM source archive/version; refusing unverified fallback provenance (SHA256=${RAGGED_LLVM_SOURCE_ARCHIVE_SHA256})")
  endif()
  set(RAGGED_VERIFIED_LLVM_REVISION "0d261d1ca552c95a8f007e061c787ac7132fbcbc")
endif()

set(RAGGED_HAS_LLVM_REVISION_HEADER FALSE)
foreach(include_dir IN LISTS LLVM_INCLUDE_DIRS)
  set(revision_header "${include_dir}/llvm/Support/VCSRevision.h")
  if(EXISTS "${revision_header}")
    file(STRINGS "${revision_header}" revision_definition
         REGEX "^[ \t]*#[ \t]*define[ \t]+LLVM_REVISION[ \t]+")
    if(revision_definition)
      set(RAGGED_HAS_LLVM_REVISION_HEADER TRUE)
    endif()
    # Match the compiler's include search order.
    break()
  endif()
endforeach()
if(NOT RAGGED_HAS_LLVM_REVISION_HEADER AND NOT RAGGED_VERIFIED_LLVM_REVISION)
  message(FATAL_ERROR "LLVM SDK does not define LLVM_REVISION. Set RAGGED_LLVM_SOURCE_ARCHIVE to the verified official llvm-project-23.1.3.src.tar.xz used to build this SDK; unidentifiable LLVM trees are not accepted.")
endif()
