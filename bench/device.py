"""Read CUDA Driver provenance outside measured calls; never infer cache hits."""
from __future__ import annotations

import ctypes
import re
import shutil
import subprocess


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
