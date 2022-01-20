workspace "neural-network"

    location "build"
    
    -- used for storing compiler / linker settings togehter
    configurations { "Debug", "Release" }

    -- enable symbols for Debug builds
    filter "configurations:Debug"
        defines "DEBUG"
        buildoptions { "-g" }
        symbols "On"

    -- enable optimization for Release builds
    filter "configurations:Release"
        defines "NDEBUG"
        optimize "On"

    -- reset filter
    filter { }


    project "neural-network"

        -- architecture: 'x86' or 'x86_64'
        architecture "x86_64"

        staticruntime "on"
    
        -- language to be compiled and c++ flavor
        language "C++"
        cppdialect "C++17"

        -- set flags for the compiler
        flags { "MultiProcessorCompile" }

        -- used for storing compiler / linker settings togehter
        configurations { "Debug", "Release" }

        -- enable symbols for Debug builds
        filter "configurations:Debug"
            defines "DEBUG"
            buildoptions { "-g" }
            symbols "On"

        -- enable optimization for Release builds
        filter "configurations:Release"
            defines "NDEBUG"
            optimize "On"

        -- reset filter
        filter { }

        kind "ConsoleApp"

        outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
        
        targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
        objdir ("%{wks.location}/obj/" .. outputdir .. "/%{prj.name}")

        files
        {
            "src/**.cpp",
            "src/**.hpp",
            "src/**.h",
        }

        defines
        {
        }

        includedirs
        {
            "src/Eigen/",
        }

        libdirs
        {
        }

        links
        {
            "bz2",          -- dep of freetype2
            "png",          -- dep of freetype2
            "z",            -- zlib, dep of assmip
            "pthread",      -- for lots of stuff
            "dl",           -- dep of glfw
            "X11",          -- dep of glfw (Linux only)
        }
        

        filter "configurations.Debug"
            runtime "Debug"

        filter "configurations.Release"
            runtime "Release"
        
