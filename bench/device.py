"""Read CUDA Driver provenance outside measured calls; never infer cache hits."""
from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import time


def cuda_driver_info(ordinal: int) -> dict:
    result = {"device_ordinal": ordinal, "l2_cache_size_bytes": None,
              "driver_package_version": None, "device_uuid_hex": None}
    executable = shutil.which("nvidia-smi")
    if executable:
        command = [executable, "--query-gpu=driver_version", "--format=csv,noheader"]
        try:
            probe = subprocess.run(command, capture_output=True, text=True, timeout=30)
            result["driver_package_probe"] = {"command": command, "exit_code": probe.returncode,
                                               "stdout": probe.stdout, "stderr": probe.stderr}
            versions = set(probe.stdout.strip().splitlines())
            if (probe.returncode == 0 and len(versions) == 1
                    and re.fullmatch(r"\d+\.\d+(?:\.\d+)?", next(iter(versions)) or "")):
                result["driver_package_version"] = next(iter(versions))
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["driver_package_probe"] = {"command": command, "error": str(exc)}
    else:
        result["driver_package_probe"] = {"available": False}
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDriverGetVersion.restype = ctypes.c_int
        version = ctypes.c_int()
        result["cuda_supported_version_status"] = driver.cuDriverGetVersion(ctypes.byref(version))
        if result["cuda_supported_version_status"] == 0:
            # This is the maximum CUDA version supported by the Driver API,
            # not the NVIDIA Linux driver package/build version.
            result["cuda_supported_version"] = version.value
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        result["cuInit_status"] = driver.cuInit(0)
        if result["cuInit_status"] != 0:
            return result
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGet.restype = ctypes.c_int
        device = ctypes.c_int()
        result["device_get_status"] = driver.cuDeviceGet(ctypes.byref(device), ordinal)
        if result["device_get_status"] != 0:
            return result
        uuid_function = getattr(driver, "cuDeviceGetUuid_v2", None)
        if uuid_function is not None:
            uuid = (ctypes.c_ubyte * 16)()
            uuid_function.argtypes = [ctypes.c_void_p, ctypes.c_int]
            uuid_function.restype = ctypes.c_int
            result["uuid_query_status"] = uuid_function(ctypes.byref(uuid), device.value)
            if result["uuid_query_status"] == 0:
                result["device_uuid_hex"] = bytes(uuid).hex()
        driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        driver.cuDeviceGetAttribute.restype = ctypes.c_int
        cache = ctypes.c_int()
        # CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE=38 in the pinned CUDA 13.2 cuda.h.
        result["l2_query_status"] = driver.cuDeviceGetAttribute(ctypes.byref(cache), 38, device.value)
        if result["l2_query_status"] == 0 and cache.value > 0:
            result["l2_cache_size_bytes"] = cache.value
        return result
    except (OSError, AttributeError) as exc:
        result["error"] = str(exc)
        return result


GPU_STATE_FIELDS = ("index", "uuid", "name", "pstate", "clocks.sm", "clocks.mem", "clocks.max.sm",
                    "clocks.max.mem", "temperature.gpu", "power.draw", "power.limit", "utilization.gpu",
                    "utilization.memory", "memory.used", "clocks_event_reasons.active")


def _query(command):
    try:
        probe = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if probe.returncode:
        return None, probe.stderr.strip() or f"exit {probe.returncode}"
    return [[value.strip() for value in line.split(",")] for line in probe.stdout.strip().splitlines()], None


def gpu_state() -> dict:
    """Clocks, temperature, power, utilization and other compute processes, from nvidia-smi.

    Clocks cannot be locked without administrator rights, so each run records
    them instead. The host load average is included because host launch gaps
    count in the complete-call times.
    """
    state = {"time_unix_ns": time.time_ns(), "host_loadavg": list(os.getloadavg())}
    executable = shutil.which("nvidia-smi")
    if not executable:
        state["error"] = "nvidia-smi not found"
        return state
    rows, error = _query([executable, "--query-gpu=" + ",".join(GPU_STATE_FIELDS), "--format=csv,noheader,nounits"])
    if error:
        state["error"] = error
    else:
        state["gpus"] = [dict(zip(GPU_STATE_FIELDS, row)) for row in rows]
    apps, error = _query([executable, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    state["compute_processes"] = ([{"pid": row[0], "name": row[1], "used_memory_mib": row[2]} for row in apps if len(row) == 3]
                                  if apps is not None else {"error": error})
    return state


def source_state(root) -> dict:
    """The git commit of the checkout and any tracked files that differ from it."""
    git = shutil.which("git")
    if not git:
        return {"git_commit": None, "error": "git not found"}
    try:
        head = subprocess.run([git, "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        status = subprocess.run([git, "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"git_commit": None, "error": str(exc)}
    if head.returncode:
        return {"git_commit": None, "error": head.stderr.strip()}
    return {"git_commit": head.stdout.strip(), "tracked_changes": status.stdout.splitlines()}
