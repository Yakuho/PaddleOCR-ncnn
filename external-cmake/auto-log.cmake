find_package(Git REQUIRED)
include(FetchContent)

set(FETCHCONTENT_SOURCE_DIR_EXTERN_AUTOLOG ${CMAKE_CURRENT_BINARY_DIR}/third-party/extern_autolog-src)
set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/third-party)

FetchContent_Declare(
        extern_autolog
        PREFIX autolog
        GIT_REPOSITORY https://github.com/LDOUBLEV/AutoLog.git
        GIT_TAG        main
)
FetchContent_MakeAvailable(extern_autolog)
