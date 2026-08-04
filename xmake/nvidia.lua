target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")

    local cuda = get_config("cuda")
    if cuda then
        add_includedirs(path.join(cuda, "include"))
        if is_plat("windows") then
            add_linkdirs(path.join(cuda, "lib", "x64"))
        else
            add_linkdirs(path.join(cuda, "lib64"))
        end
    end
    add_links("cudart")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/nvidia/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("cuda")
    set_policy("build.cuda.devlink", true)
    if is_plat("windows") then
        set_runtimes("MD")
    end
    add_cugencodes(get_config("cuda-arch") or "sm_86")

    local cuda = get_config("cuda")
    if cuda then
        add_includedirs(path.join(cuda, "include"))
        if is_plat("windows") then
            add_linkdirs(path.join(cuda, "lib", "x64"))
        else
            add_linkdirs(path.join(cuda, "lib64"))
        end
    end
    add_links("cudart")
    if is_plat("windows") then
        add_cuflags("-Xcompiler=/MD")
    else
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
    end

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
