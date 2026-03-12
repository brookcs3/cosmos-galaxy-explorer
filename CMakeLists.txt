cmake_minimum_required(VERSION 3.16)
project(CosmosGalaxyExplorer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- ImGui Sources (vendored) ---
set(IMGUI_SOURCES
    lib/imgui.cpp
    lib/imgui_draw.cpp
    lib/imgui_tables.cpp
    lib/imgui_widgets.cpp
    lib/imgui_impl_glfw.cpp
    lib/imgui_impl_opengl3.cpp
)

# --- Main Executable ---
add_executable(cosmos sample.cpp ${IMGUI_SOURCES})
target_include_directories(cosmos PRIVATE lib)

# --- Platform-Specific Linking ---
if(APPLE)
    # macOS: use Homebrew or system GLFW/GLEW, link frameworks
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(OpenGL REQUIRED)

    target_link_libraries(cosmos PRIVATE
        glfw
        GLEW::GLEW
        OpenGL::GL
        "-framework CoreAudio"
        "-framework AudioToolbox"
        "-framework CoreFoundation"
    )

    # GLM (header-only, just need include path)
    find_path(GLM_INCLUDE_DIR glm/glm.hpp HINTS /opt/homebrew/include /usr/local/include)
    if(GLM_INCLUDE_DIR)
        target_include_directories(cosmos PRIVATE ${GLM_INCLUDE_DIR})
    endif()

    # Suppress deprecation warnings for OpenGL on macOS
    target_compile_definitions(cosmos PRIVATE GL_SILENCE_DEPRECATION)

elseif(WIN32)
    # Windows: expect GLFW and GLEW in deps/ or find via cmake
    # If using prebuilt deps (from build script), set paths:
    if(DEFINED GLFW_DIR)
        target_include_directories(cosmos PRIVATE ${GLFW_DIR}/include)
        target_link_directories(cosmos PRIVATE ${GLFW_DIR}/lib)
    endif()
    if(DEFINED GLEW_DIR)
        target_include_directories(cosmos PRIVATE ${GLEW_DIR}/include)
        target_link_directories(cosmos PRIVATE ${GLEW_DIR}/lib/Release/x64)
    endif()
    if(DEFINED GLM_DIR)
        target_include_directories(cosmos PRIVATE ${GLM_DIR})
    endif()

    target_link_libraries(cosmos PRIVATE
        glfw3
        glew32
        opengl32
        gdi32
        user32
        shell32
        comdlg32
    )

    # Static runtime for portability
    set_property(TARGET cosmos PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

else()
    # Linux
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(OpenGL REQUIRED)

    target_link_libraries(cosmos PRIVATE
        glfw
        GLEW::GLEW
        OpenGL::GL
        dl
        pthread
    )

    find_path(GLM_INCLUDE_DIR glm/glm.hpp)
    if(GLM_INCLUDE_DIR)
        target_include_directories(cosmos PRIVATE ${GLM_INCLUDE_DIR})
    endif()
endif()

# --- Compiler Warnings ---
target_compile_options(cosmos PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wno-deprecated -Wno-unused-function>
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4996>
)

# --- Copy runtime resources next to binary ---
add_custom_command(TARGET cosmos POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:cosmos>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:cosmos>/data
)
