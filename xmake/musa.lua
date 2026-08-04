local musa = get_config("musa-path") or "/usr/local/musa"
local musa_include = path.join(musa, "include")
local musa_lib = path.join(musa, "lib")
local arch = get_config("musa-arch") or "mp_31"

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_defines("__MUSACC__")
    add_includedirs(musa_include)
    add_linkdirs(musa_lib)
    add_links("musart")
    add_syslinks("dl")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", "-Wno-error=attributes")
    end
    add_files("../src/device/nvidia/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    -- Tell xmake to use its known g++ command-line rules with mcc as the program.
    set_toolset("cxx", "g++@" .. path.join(os.projectdir(), "xmake", "musa-g++"))
    add_includedirs(musa_include)
    add_linkdirs(musa_lib)
    add_links("musart")
    add_syslinks("dl")
    add_cxflags(
        "-mtgpu",
        "--musa-path=" .. musa,
        "--offload-arch=" .. arch,
        {force = true})
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("../src/ops/*/nvidia/*.cu", {sourcekind = "cxx"})

    on_install(function (target) end)
target_end()
