add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

option("cuda-arch")
    set_default("sm_86")
    set_showmenu(true)
    set_description("CUDA GPU architecture used for Nvidia kernels")
option_end()

option("musa-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile the Moore Threads MUSA backend")
option_end()

option("musa-path")
    set_default("/usr/local/musa")
    set_showmenu(true)
    set_description("MUSA SDK directory")
option_end()

option("musa-arch")
    set_default("mp_31")
    set_showmenu(true)
    set_description("MUSA GPU architecture used for kernels")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

if has_config("musa-gpu") then
    add_defines("ENABLE_NVIDIA_API", "ENABLE_MUSA_API")
    includes("xmake/musa.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("nv-gpu") or has_config("musa-gpu") then
        add_deps("llaisys-device-nvidia")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    if has_config("musa-gpu") then
        add_syslinks("dl")
    end

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    if has_config("nv-gpu") or has_config("musa-gpu") then
        add_deps("llaisys-ops-nvidia")
    end

    if has_config("musa-gpu") then
        add_linkdirs(path.join(get_config("musa-path") or "/usr/local/musa", "lib"))
        add_links("musart")
        add_syslinks("dl")
        set_toolset(
            "sh",
            "g++@" .. path.join(os.projectdir(), "xmake", "musa-g++"))
        add_shflags(
            "-mtgpu",
            "--musa-path=" .. (get_config("musa-path") or "/usr/local/musa"),
            "--offload-arch=" .. (get_config("musa-arch") or "mp_31"))
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()
