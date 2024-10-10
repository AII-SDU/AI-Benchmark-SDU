'''
Copyright (c) 2024, 山东大学智能创新研究院(Academy of Intelligent Innovation)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
'''
# Copyright (c) Academy of Intelligent Innovation.
# License-Identifier: BSD 2-Clause License
# AI Benchmark SDU Team

import ctypes
import inspect
from ctypes import *
from enum import Enum
from collections import namedtuple

### Constant Definitions
MTML_LIBRARY_VERSION_BUFFER_SIZE = 32
MTML_DRIVER_VERSION_BUFFER_SIZE = 80
MTML_DEVICE_NAME_BUFFER_SIZE = 32
MTML_DEVICE_UUID_BUFFER_SIZE = 48
MTML_DEVICE_MTBIOS_VERSION_BUFFER_SIZE = 64
MTML_DEVICE_VBIOS_VERSION_BUFFER_SIZE = MTML_DEVICE_MTBIOS_VERSION_BUFFER_SIZE
MTML_DEVICE_PATH_BUFFER_SIZE = 64
MTML_DEVICE_PCI_SBDF_BUFFER_SIZE = 32
MTML_VIRT_TYPE_ID_BUFFER_SIZE = 16
MTML_VIRT_TYPE_CLASS_BUFFER_SIZE = 32
MTML_VIRT_TYPE_NAME_BUFFER_SIZE = 32
MTML_VIRT_TYPE_API_BUFFER_SIZE = 16
MTML_DEVICE_PCI_BUS_ID_FMT = "%08X:%02X:%02X.0"
MTML_LOG_FILE_PATH_BUFFER_SIZE = 200
MTML_MPC_PROFILE_NAME_BUFFER_SIZE = 32
MTML_MPC_CONF_NAME_BUFFER_SIZE = 32
MTML_MPC_CONF_MAX_PROF_NUM = 16
MTML_DEVICE_SLOT_NAME_BUFFER_SIZE = 32
MTML_MEMORY_VENDOR_BUFFER_SIZE = 64
MTML_DEVICE_SERIAL_NUMBER_BUFFER_SIZE = 64

### Enumerations
class MtmlReturn(Enum):
    MTML_SUCCESS = 0
    MTML_ERROR_DRIVER_NOT_LOADED = 1
    MTML_ERROR_DRIVER_FAILURE = 2
    MTML_ERROR_INVALID_ARGUMENT = 3
    MTML_ERROR_NOT_SUPPORTED = 4
    MTML_ERROR_NO_PERMISSION = 5
    MTML_ERROR_INSUFFICIENT_SIZE = 6
    MTML_ERROR_NOT_FOUND = 7
    MTML_ERROR_INSUFFICIENT_MEMORY = 8
    MTML_ERROR_DRIVER_TOO_OLD = 9
    MTML_ERROR_DRIVER_TOO_NEW = 10
    MTML_ERROR_TIMEOUT = 11
    MTML_ERROR_UNKNOWN = 999

class MtmlBrandType(Enum):
    MTML_BRAND_MTT = 0
    MTML_BRAND_UNKNOWN = 1
    MTML_BRAND_COUNT = 2

class MtmlMemoryType(Enum):
    MTML_MEM_TYPE_LPDDR4 = 0
    MTML_MEM_TYPE_GDDR6 = 1

class MtmlCodecType (Enum):
    MTML_CODEC_TYPE_AVC = 0 
    MTML_CODEC_TYPE_VC1 = 1 
    MTML_CODEC_TYPE_MPEG2 = 2 
    MTML_CODEC_TYPE_MPEG4 = 3 
    MTML_CODEC_TYPE_H263 = 4 
    MTML_CODEC_TYPE_DIV3 = 5 
    MTML_CODEC_TYPE_RV = 6 
    MTML_CODEC_TYPE_AVS = 7 
    MTML_CODEC_TYPE_RSVD1 = 8 
    MTML_CODEC_TYPE_THO = 9 
    MTML_CODEC_TYPE_VP3 = 10 
    MTML_CODEC_TYPE_VP8 = 11 
    MTML_CODEC_TYPE_HEVC = 12 
    MTML_CODEC_TYPE_VP9 = 13 
    MTML_CODEC_TYPE_AVS2 = 14 
    MTML_CODEC_TYPE_RSVD2 = 15 
    MTML_CODEC_TYPE_AV1 = 16 
    MTML_CODEC_TYPE_COUNT = 17

class MtmlCodecSessionState(Enum):
    MTML_CODEC_SESSION_STATE_UNKNOWN = -1
    MTML_CODEC_SESSION_STATE_IDLE = 0
    MTML_CODEC_SESSION_STATE_ACTIVE = 1
    MTML_CODEC_SESSION_STATE_COUNT = 2
    
class MtmlVirtCapability (Enum):
    MTML_DEVICE_NOT_SUPPORT_VIRTUALIZATION = 0 
    MTML_DEVICE_SUPPORT_VIRTUALIZATION = 1

class MtmlVirtRole(Enum):
    MTML_VIRT_ROLE_NONE = 0
    MTML_VIRT_ROLE_HOST_VIRTDEVICE = 1
    MTML_VIRT_ROLE_COUNT = 2

class MtmlDeviceTopologyLevel(Enum):
    MTML_TOPOLOGY_INTERNAL = 0
    MTML_TOPOLOGY_SINGLE = 1
    MTML_TOPOLOGY_MULTIPLE = 2
    MTML_TOPOLOGY_HOSTBRIDGE = 3
    MTML_TOPOLOGY_NODE = 4
    MTML_TOPOLOGY_SYSTEM = 5

class MtmlLogLevel(Enum):
    MTML_LOG_LEVEL_OFF = 0
    MTML_LOG_LEVEL_FATAL = 1
    MTML_LOG_LEVEL_ERROR = 2
    MTML_LOG_LEVEL_WARNING = 3
    MTML_LOG_LEVEL_INFO = 4

class MtmlMpcMode(Enum):
    MTML_DEVICE_MPC_DISABLE = 0
    MTML_DEVICE_MPC_ENABLE = 1

class MtmlMpcCapability(Enum):
    MTML_DEVICE_NOT_SUPPORT_MPC = 0
    MTML_DEVICE_SUPPORT_MPC = 1

class MtmlMpcType(Enum):
    MTML_MPC_TYPE_NONE = 0
    MTML_MPC_TYPE_PARENT = 1
    MTML_MPC_TYPE_INSTANCE = 2

class MtmlDeviceP2PStatus(Enum):
    MTML_P2P_STATUS_OK = 0
    MTML_P2P_STATUS_CHIPSET_NOT_SUPPORTED = 1
    MTML_P2P_STATUS_GPU_NOT_SUPPORTED = 2
    MTML_P2P_STATUS_UNKNOWN = 3

class MtmlDeviceP2PCaps(Enum):
    MTML_P2P_CAPS_READ = 0
    MTML_P2P_CAPS_WRITE = 1

class MtmlMtLinkState(Enum):
    MTML_MTLINK_STATE_DOWN = 0
    MTML_MTLINK_STATE_UP = 1

class MtmlMtLinkCap(Enum):
    MTML_MTLINK_CAP_P2P_ACCESS = 0
    MTML_MTLINK_CAP_P2P_ATOMICS = 1
    MTML_MTLINK_CAP_COUNT = 2

class MtmlMtLinkCapStatus(Enum):
    MTML_MTLINK_CAP_STATUS_NOT_SUPPORTED = 0
    MTML_MTLINK_CAP_STATUS_OK = 1

class MtmlMtLinkCapability(Enum):
    MTML_DEVICE_NOT_SUPPORT_MTLINK = 0
    MTML_DEVICE_SUPPORT_MTLINK = 1

class MtmlDispIntfType(Enum):
    MTML_DISP_INTF_TYPE_DP = 0
    MTML_DISP_INTF_TYPE_EDP = 1
    MTML_DISP_INTF_TYPE_VGA = 2
    MTML_DISP_INTF_TYPE_HDMI = 3
    MTML_DISP_INTF_TYPE_LVDS = 4
    MTML_DISP_INTF_TYPE_MAX = 5

class MtmlGpuEngine(Enum):
    MTML_GPU_ENGINE_GEOMETRY = 0
    MTML_GPU_ENGINE_2D = 1
    MTML_GPU_ENGINE_3D = 2
    MTML_GPU_ENGINE_COMPUTE = 3
    MTML_GPU_ENGINE_MAX = 4

class MtmlEccMode(Enum):
    MTML_MEMORY_ECC_DISABLE = 0
    MTML_MEMORY_ECC_ENABLE = 1

class MtmlPageRetirementCause(Enum):
    MTML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS = 0
    MTML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR = 1
    MTML_PAGE_RETIREMENT_CAUSE_MAX = 2

class MtmlRetiredPagesPendingState(Enum):
    MTML_RETIRED_PAGES_PENDING_STATE_FALSE = 0
    MTML_RETIRED_PAGES_PENDING_STATE_TRUE = 1
    
### Load MTML
_lib = ctypes.CDLL("libmtml.so")

### Error Process
# class MTMLError(Exception):
#     _errcode_to_string = {
#         MtmlReturn.MTML_SUCCESS: "Success",
#         MtmlReturn.MTML_ERROR_DRIVER_NOT_LOADED: "Driver Not Loaded",
#         MtmlReturn.MTML_ERROR_DRIVER_FAILURE: "Driver Failure",
#         MtmlReturn.MTML_ERROR_INVALID_ARGUMENT: "Invalid Argument",
#         MtmlReturn.MTML_ERROR_NOT_SUPPORTED: "Not Supported",
#         MtmlReturn.MTML_ERROR_NO_PERMISSION: "Insufficient Permissions",
#         MtmlReturn.MTML_ERROR_INSUFFICIENT_SIZE: "Insufficient Size",
#         MtmlReturn.MTML_ERROR_NOT_FOUND: "Not Found",
#         MtmlReturn.MTML_ERROR_INSUFFICIENT_MEMORY: "Insufficient Memory",
#         MtmlReturn.MTML_ERROR_DRIVER_TOO_OLD: "Driver Too Old",
#         MtmlReturn.MTML_ERROR_DRIVER_TOO_NEW: "Driver Too New",
#         MtmlReturn.MTML_ERROR_TIMEOUT: "Timeout",
#         MtmlReturn.MTML_ERROR_UNKNOWN: "Unknown Error",
#     }

#     def __new__(cls, value):
#         if not isinstance(value, MtmlReturn):
#             raise TypeError("Expected an MtmlReturn enum member")
#         if value not in cls._errcode_to_string:
#             raise ValueError("Unknown error code")
#         return super(MTMLError, cls).__new__(cls)

#     def __init__(self, value):
#         self.value = value

#     def __str__(self):
#         return self._errcode_to_string[self.value]

# def _mtmlErrorCheck(ret):
#     if ret != MtmlReturn.MTML_SUCCESS.value:
#         raise MTMLError(MtmlReturn(ret))

def _mtmlErrorCheck(ret):
    func_name = inspect.currentframe().f_back.f_code.co_name
    error_message = mtmlErrorString(ret)
    if error_message != 'Success':
        print(f"Error in {func_name}(): {error_message}")

### Structure
class _PrintableStructure(Structure):
    _fmt_ = {"<default>": "%s"}

    def __str__(self):
        result = []
        for field in self._fields_:
            field_name, field_type = field[0], field[1]
            value = getattr(self, field_name)
            fmt = self._fmt_.get(field_name, self._fmt_.get("<default>", "%s"))
            result.append(f"{field_name}: {fmt % value}")
        return self.__class__.__name__ + "(" + ", ".join(result) + ")"

class MtmlPciInfo(_PrintableStructure):
    _fields_ = [
        ("sbdf", c_char * MTML_DEVICE_PCI_SBDF_BUFFER_SIZE),  # PCI identifier
        ("segment", c_uint),  # PCI segment
        ("bus", c_uint),      # Bus
        ("device", c_uint),   # Device ID
        ("pciDeviceId", c_uint),  # Combined device ID and vendor ID
        ("pciSubsystemId", c_uint),  # Subsystem ID
        ("busWidth", c_uint),        # Bus width
        ("pciMaxSpeed", c_float),    # Maximum link speed (GT/s)
        ("pciCurSpeed", c_float),     # Current link speed (GT/s)
        ("pciMaxWidth", c_uint),      # Maximum link width
        ("pciCurWidth", c_uint),      # Current link width
        ("pciMaxGen", c_uint),        # Maximum supported generation
        ("pciCurGen", c_uint),        # Current generation
        ("rsvd", c_uint * 6)          # Reserved field
    ]
    
class MtmlPciSlotInfo(_PrintableStructure):
    _fields_ = [
        ("slotId", c_uint),                          # Unique ID of the PCI slot
        ("slotName", c_char * MTML_DEVICE_SLOT_NAME_BUFFER_SIZE),  # PCI slot name
        ("rsvd", c_uint * 4)                         # Reserved field
    ]

class MtmlCodecUtil(_PrintableStructure):
    _fields_ = [
        ("util", c_uint),   # Overall utilization of the codec
        ("period", c_uint), # Sampling period (microseconds)
        ("encUtil", c_uint), # Encoder utilization
        ("decUtil", c_uint), # Decoder utilization
        ("rsvd", c_int * 2)  # Reserved field
    ]
     
class MtmlCodecSessionMetrics(_PrintableStructure):
    _fields_ = [
        ("id", c_uint),              # Unique identifier of the encoding session
        ("pid", c_uint),             # Process ID, 0 indicates an idle session
        ("hResolution", c_uint),     # Horizontal resolution
        ("vResolution", c_uint),     # Vertical resolution
        ("frameRate", c_uint),       # Frame rate (frames per second)
        ("bitRate", c_uint),         # Bit rate (bits per second)
        ("latency", c_uint),         # Codec latency (microseconds)
        ("codecType", c_uint),       # Codec type, e.g., H.265 and AV1
        ("rsvd", c_int * 4)          # Reserved field
    ]

class MtmlVirtType(_PrintableStructure):
    _fields_ = [
        ("id", c_char * MTML_VIRT_TYPE_ID_BUFFER_SIZE),        # ID of the virtualization type
        ("name", c_char * MTML_VIRT_TYPE_NAME_BUFFER_SIZE),    # Name of the virtualization type
        ("api", c_char * MTML_VIRT_TYPE_API_BUFFER_SIZE),      # API type of the virtualization
        ("horizontalResolution", c_uint),                        # Maximum X dimension pixel count
        ("verticalResolution", c_uint),                          # Maximum Y dimension pixel count
        ("frameBuffer", c_uint),                                # Frame buffer size (MB)
        ("maxEncodeNum", c_uint),                               # Maximum encoding count
        ("maxDecodeNum", c_uint),                               # Maximum decoding count
        ("maxInstances", c_uint),                               # Maximum vGPU instances per physical GPU
        ("maxVirtualDisplay", c_uint),                          # Number of display heads
        ("rsvd", c_int * 11)                                   # Reserved field
    ]

class MtmlDeviceProperty(_PrintableStructure):
    _fields_ = [
        ("virtCap", c_uint, 1),     # Virtualization capability
        ("virtRole", c_uint, 3),    # Virtualization role
        ("mpcCap", c_uint, 1),      # MPC capability
        ("mpcType", c_uint, 3),     # MPC type
        ("mtLinkCap", c_uint, 1),   # MtLink capability
        ("rsvd", c_uint, 23),       # Reserved field
        ("rsvd2", c_uint, 32)       # Reserved field
    ]

## struct MtmlLogConfiguration 
class MtmlConsoleConfig(_PrintableStructure):
    _fields_ = [
        ("level", c_uint),  # Log level
        ("rsvd", c_int * 2) # Reserved field
    ]

class MtmlSystemConfig(_PrintableStructure):
    _fields_ = [
        ("level", c_uint),  # Log level
        ("rsvd", c_int * 2) # Reserved field
    ]

class MtmlFileConfig(_PrintableStructure):
    _fields_ = [
        ("level", c_uint),  # Log level
        ("file", c_char * 200),  # Log file path
        ("size", c_uint),  # Maximum log file size
        ("rsvd", c_int * 2) # Reserved field
    ]

class MtmlCallbackConfig(_PrintableStructure):
    _fields_ = [
        ("level", c_uint),  # Log level
        ("callback", CFUNCTYPE(None, c_char_p, c_uint)),  # Callback function
        ("rsvd", c_int * 2) # Reserved field
    ]

class MtmlLogConfiguration(_PrintableStructure):
    _fields_ = [
        ("consoleConfig", MtmlConsoleConfig),
        ("systemConfig", MtmlSystemConfig),
        ("fileConfig", MtmlFileConfig),
        ("callbackConfig", MtmlCallbackConfig)
    ]

class MtmlMpcProfile(_PrintableStructure):
    _fields_ = [
        ("id", c_uint),  # Profile ID within the device
        ("coreCount", c_uint),  # MPC core count
        ("memorySizeMB", c_ulonglong),  # Memory size in MBytes
        ("name", c_char * MTML_MPC_PROFILE_NAME_BUFFER_SIZE),  # Profile name
        ("rsvd", c_uint * 10)  # Reserved field
    ]
    
class MtmlMpcConfiguration(_PrintableStructure):
    _fields_ = [
        ("id", c_uint),  # Configuration ID
        ("name", c_char * MTML_MPC_CONF_NAME_BUFFER_SIZE),  # Configuration name
        ("rsvd", c_uint * 24)  # Reserved field
    ]

class MtmlMtLinkSpec(_PrintableStructure):
    _fields_ = [
        ("version", c_uint),  # Version of MtLink
        ("bandWidth", c_uint),  # Bandwidth per link in GB/s
        ("linkNum", c_uint),  # Maximum number of supported links
        ("rsvd", c_uint * 4)  # Reserved for future extensions
    ]

class MtmlMtLinkLayout(_PrintableStructure):
    _fields_ = [
        ("localLinkId", c_uint),  # Local link ID
        ("remoteLinkId", c_uint),  # Remote link ID
        ("rsvd", c_uint * 4)  # Reserved for future extensions
    ]

class MtmlDispIntfSpec(_PrintableStructure):
    _fields_ = [
        ("type", c_uint),  # Display interface type
        ("maxHoriRes", c_uint),  # Maximum horizontal resolution
        ("maxVertRes", c_uint),  # Maximum vertical resolution
        ("maxRefreshRate", c_float),  # Maximum refresh rate
        ("rsvd", c_uint * 8)  # Reserved for future extension
    ]
    
class MtmlPageRetirementCount(_PrintableStructure):
    _fields_ = [
        ("sbeCount", c_uint),  # Single bit ECC error count
        ("dbeCount", c_uint),  # Double bit ECC error count
        ("rsvd", c_uint * 10)  # Reserved for future extension
    ]

class MtmlPageRetirement(_PrintableStructure):
    _fields_ = [
        ("cause", c_uint),  # Cause of page retirement
        ("timestamps", c_ulonglong),  # Timestamps when page retirement occurred
        ("address", c_ulonglong),  # Physical address of the retired page
        ("rsvd", c_uint * 10)  # Reserved for future extension
    ]

class MtmlPageRetirementPending(_PrintableStructure):
    _fields_ = [
        ("cause", c_uint),  # Cause of pending page retirement
        ("timestamps", c_ulonglong),  # Timestamps when pending retirement started
        ("address", c_ulonglong),  # Physical address of the page awaiting retirement
        ("rsvd", c_uint * 10)  # Reserved for future extension
    ]

### Declare opaque handle pointer
'''
- MtmlLibrary represents the MTML library itself and plays as the entry point to access and create other opaque objects.
- MtmlSystem represents the system environment in which the library is running.
- MtmlDevice represents the Moore Threads device (including virtual devices) that is installed in the sys-tem
- MtmlGpu represents the graphic unit of a Moore Threads device, which is responsible for the 3D and compute workloads.
- MtmlMemory represents the memory units that reside on a Moore Threads device.
- MtmlVpu represents the video codec unit of a Moore Threads device which handles the video encodin-g decoding task.

    The relationship among the above opaque data types is hierarchical, which means some type 'contains' other types. 
    The hierarchy of opaque data types mentioned above can be summarized as follows.

    Library
    |--- System
    |--- Device
            |--- GPU
            |--- Memory
            |--- VPU
'''
## MtmlLibrary
class struct_c_mtmlLibrary_t(Structure):
    pass
c_mtmlLibrary_t = POINTER(struct_c_mtmlLibrary_t)()

## MtmlSystem
class struct_c_mtmlSystem_t(Structure):
    pass
c_mtmlSystem_t = POINTER(struct_c_mtmlSystem_t)()

## DeviceSystem
class struct_c_mtmlDevice_t(Structure):
    pass
c_mtmlDevice_t = POINTER(struct_c_mtmlDevice_t)()

## MtmlGpu
class struct_c_mtmlGpu_t(Structure):
    pass
c_mtmlGpu_t = POINTER(struct_c_mtmlGpu_t)()

## MtmlMemory
class struct_c_mtmlMemory_t(Structure):
    pass
c_mtmlMemory_t = POINTER(struct_c_mtmlMemory_t)()

## MtmlVpu
class struct_c_mtmlVpu_t(Structure):
    pass
c_mtmlVpu_t = POINTER(struct_c_mtmlVpu_t)()

### Functions
## Library Functions
def mtmlLibraryInit(lib):
    fn = _lib.mtmlLibraryInit
    fn.restype = c_int
    result = fn(byref(lib))
    _mtmlErrorCheck(result)
    return lib

def mtmlLibraryShutDown(lib):
    fn = _lib.mtmlLibraryShutDown
    fn.restype = c_int
    result = fn(lib)
    _mtmlErrorCheck(result)

def mtmlLibraryGetVersion(lib):
    version = create_string_buffer(MTML_LIBRARY_VERSION_BUFFER_SIZE)
    length = c_uint(MTML_LIBRARY_VERSION_BUFFER_SIZE)
    fn = _lib.mtmlLibraryGetVersion
    fn.restype = c_int
    result = fn(lib, version, length)
    _mtmlErrorCheck(result)
    return version.value.decode()

def mtmlLibraryInitSystem(lib, sys):
    fn = _lib.mtmlLibraryInitSystem
    fn.restype = c_int
    result = fn(lib, byref(sys))
    _mtmlErrorCheck(result)
    return sys

def mtmlLibraryFreeSystem(sys):
    fn = _lib.mtmlLibraryFreeSystem
    fn.restype = c_int
    result = fn(sys)
    _mtmlErrorCheck(result)

def mtmlLibraryCountDevice(lib):
    count = c_uint()
    fn = _lib.mtmlLibraryCountDevice
    fn.restype = c_int
    result = fn(lib, byref(count))
    _mtmlErrorCheck(result)
    return count.value

def mtmlLibraryInitDeviceByIndex(lib, index, dev):
    fn = _lib.mtmlLibraryInitDeviceByIndex
    fn.restype = c_int
    result = fn(lib, index, byref(dev))
    _mtmlErrorCheck(result)
    return dev

def mtmlLibraryInitDeviceByUuid(lib, uuid, dev):
    fn = _lib.mtmlLibraryInitDeviceByUuid
    fn.restype = c_int
    result = fn(lib, uuid.encode(), byref(dev))
    _mtmlErrorCheck(result)
    return dev

def mtmlLibraryInitDeviceByPciSbdf(lib, pciSbdf, dev):
    fn = _lib.mtmlLibraryInitDeviceByPciSbdf
    fn.restype = c_int
    result = fn(lib, pciSbdf.encode(), byref(dev))
    _mtmlErrorCheck(result)
    return dev

def mtmlLibrarySetMpcConfigurationInBatch(lib, count, devices, mpcConfigIds):
    fn = _lib.mtmlLibrarySetMpcConfigurationInBatch
    fn.restype = c_int
    result = fn(lib, count, devices, mpcConfigIds)
    _mtmlErrorCheck(result)

def mtmlLibraryFreeDevice(dev):
    fn = _lib.mtmlLibraryFreeDevice
    fn.restype = c_int
    result = fn(dev)
    _mtmlErrorCheck(result)

## System Functions
def mtmlSystemGetDriverVersion(sys):
    version = create_string_buffer(MTML_DRIVER_VERSION_BUFFER_SIZE)
    length = c_uint(MTML_DRIVER_VERSION_BUFFER_SIZE)
    fn = _lib.mtmlSystemGetDriverVersion
    fn.restype = c_int
    result = fn(sys, version, length)
    _mtmlErrorCheck(result)
    return version.value.decode()

## Device Functions
def mtmlDeviceInitGpu(dev, gpu):
    fn = _lib.mtmlDeviceInitGpu
    fn.restype = c_int
    result = fn(dev, byref(gpu))
    _mtmlErrorCheck(result)
    return gpu

def mtmlDeviceFreeGpu(gpu):
    fn = _lib.mtmlDeviceFreeGpu
    fn.restype = c_int
    result = fn(gpu)
    _mtmlErrorCheck(result)

def mtmlDeviceInitMemory(dev, mem):
    fn = _lib.mtmlDeviceInitMemory
    fn.restype = c_int
    result = fn(dev, byref(mem))
    _mtmlErrorCheck(result)
    return mem

def mtmlDeviceFreeMemory(mem):
    fn = _lib.mtmlDeviceFreeMemory
    fn.restype = c_int
    result = fn(mem)
    _mtmlErrorCheck(result)

def mtmlDeviceInitVpu(dev, vpu):
    fn = _lib.mtmlDeviceInitVpu
    fn.restype = c_int
    result = fn(dev, byref(vpu))
    _mtmlErrorCheck(result)
    return vpu

def mtmlDeviceFreeVpu(vpu):
    fn = _lib.mtmlDeviceFreeVpu
    fn.restype = c_int
    result = fn(vpu)
    _mtmlErrorCheck(result)

def mtmlDeviceGetIndex(dev):
    index = c_uint()
    fn = _lib.mtmlDeviceGetIndex
    fn.restype = c_int
    result = fn(dev, byref(index))
    _mtmlErrorCheck(result)
    return index.value

def mtmlDeviceGetUUID(dev):
    uuid = create_string_buffer(MTML_DEVICE_UUID_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_UUID_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetUUID
    fn.restype = c_int
    result = fn(dev, uuid, length)
    _mtmlErrorCheck(result)
    return uuid.value.decode()

def mtmlDeviceGetBrand(dev):
    brand_type = c_int()
    fn = _lib.mtmlDeviceGetBrand
    fn.restype = c_int
    result = fn(dev, byref(brand_type))
    _mtmlErrorCheck(result)
    return MtmlBrandType(brand_type.value)

def mtmlDeviceGetName(dev):
    name = create_string_buffer(MTML_DEVICE_NAME_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_NAME_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetName
    fn.restype = c_int
    result = fn(dev, name, length)
    _mtmlErrorCheck(result)
    return name.value.decode()

def mtmlDeviceGetPciInfo(dev):
    pci_info = MtmlPciInfo()
    fn = _lib.mtmlDeviceGetPciInfo
    fn.restype = c_int
    result = fn(dev, byref(pci_info))
    _mtmlErrorCheck(result)
    return pci_info

def mtmlDeviceGetPowerUsage(dev):
    power = c_uint()
    fn = _lib.mtmlDeviceGetPowerUsage
    fn.restype = c_int
    result = fn(dev, byref(power))
    _mtmlErrorCheck(result)
    return power.value

def mtmlDeviceGetGpuPath(dev):
    path = create_string_buffer(MTML_DEVICE_PATH_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_PATH_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetGpuPath
    fn.restype = c_int
    result = fn(dev, path, length)
    _mtmlErrorCheck(result)
    return path.value.decode()

def mtmlDeviceGetPrimaryPath(dev):
    path = create_string_buffer(MTML_DEVICE_PATH_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_PATH_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetPrimaryPath
    fn.restype = c_int
    result = fn(dev, path, length)
    _mtmlErrorCheck(result)
    return path.value.decode()

def mtmlDeviceGetRenderPath(dev):
    path = create_string_buffer(MTML_DEVICE_PATH_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_PATH_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetRenderPath
    fn.restype = c_int
    result = fn(dev, path, length)
    _mtmlErrorCheck(result)
    return path.value.decode()

def mtmlDeviceGetMtBiosVersion(dev):
    version = create_string_buffer(MTML_DEVICE_MTBIOS_VERSION_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_MTBIOS_VERSION_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetMtBiosVersion
    fn.restype = c_int
    result = fn(dev, version, length)
    _mtmlErrorCheck(result)
    return version.value.decode()

def mtmlDeviceGetProperty(dev):
    prop = MtmlDeviceProperty()
    fn = _lib.mtmlDeviceGetProperty
    fn.restype = c_int
    result = fn(dev, byref(prop))
    _mtmlErrorCheck(result)
    return prop

def mtmlDeviceCountFan(dev):
    count = c_uint()
    fn = _lib.mtmlDeviceCountFan
    fn.restype = c_int
    result = fn(dev, byref(count))
    _mtmlErrorCheck(result)
    return count.value

def mtmlDeviceGetFanSpeed(dev, index):
    speed = c_uint()
    fn = _lib.mtmlDeviceGetFanSpeed
    fn.restype = c_int
    result = fn(dev, index, byref(speed))
    _mtmlErrorCheck(result)
    return speed.value

def mtmlDeviceGetPcieSlotInfo(dev):
    slot_info = MtmlPciSlotInfo()
    fn = _lib.mtmlDeviceGetPcieSlotInfo
    fn.restype = c_int
    result = fn(dev, byref(slot_info))
    _mtmlErrorCheck(result)
    return slot_info

def mtmlDeviceCountDisplayInterface(dev):
    count = c_uint()
    fn = _lib.mtmlDeviceCountDisplayInterface
    fn.restype = c_int
    result = fn(dev, byref(count))
    _mtmlErrorCheck(result)
    return count.value

def mtmlDeviceGetDisplayInterfaceSpec(dev, intfIndex):
    spec = MtmlDispIntfSpec()
    fn = _lib.mtmlDeviceGetDisplayInterfaceSpec
    fn.restype = c_int
    result = fn(dev, intfIndex, byref(spec))
    _mtmlErrorCheck(result)
    return spec

def mtmlDeviceGetSerialNumber(dev):
    serial_number = create_string_buffer(MTML_DEVICE_SERIAL_NUMBER_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_SERIAL_NUMBER_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetSerialNumber
    fn.restype = c_int
    result = fn(dev, length, serial_number)
    _mtmlErrorCheck(result)
    return serial_number.value.decode()

## Device Virtualization Functions
def mtmlDeviceCountSupportedVirtTypes(dev):
    count = c_uint()
    fn = _lib.mtmlDeviceCountSupportedVirtTypes
    fn.restype = c_int
    result = fn(dev, byref(count))
    _mtmlErrorCheck(result)
    return count.value

def mtmlDeviceGetSupportedVirtTypes(dev):
    types = (MtmlVirtType * 10)()
    fn = _lib.mtmlDeviceGetSupportedVirtTypes
    fn.restype = c_int
    result = fn(dev, types, len(types))
    _mtmlErrorCheck(result)
    return [types[i] for i in range(len(types))]

def mtmlDeviceCountAvailVirtTypes(dev):
    count = c_uint()
    fn = _lib.mtmlDeviceCountAvailVirtTypes
    fn.restype = c_int
    result = fn(dev, byref(count))
    _mtmlErrorCheck(result)
    return count.value

def mtmlDeviceGetAvailVirtTypes(dev):
    types = (MtmlVirtType * 10)()
    fn = _lib.mtmlDeviceGetAvailVirtTypes
    fn.restype = c_int
    result = fn(dev, types, len(types))
    _mtmlErrorCheck(result)
    return [types[i] for i in range(len(types))]

def mtmlDeviceCountAvailVirtDevices(dev, virt_type):
    count = c_uint()
    fn = _lib.mtmlDeviceCountAvailVirtDevices
    fn.restype = c_int
    result = fn(dev, virt_type, byref(count))
    _mtmlErrorCheck(result)
    return count.value

def mtmlDeviceCountActiveVirtDevices(dev):
    count = c_uint()
    fn = _lib.mtmlDeviceCountActiveVirtDevices
    fn.restype = c_int
    result = fn(dev, byref(count))
    _mtmlErrorCheck(result)
    return count.value

def mtmlDeviceGetActiveVirtDeviceUuids(dev, entry_length, entry_count):
    uuids = create_string_buffer(entry_length * entry_count)
    fn = _lib.mtmlDeviceGetActiveVirtDeviceUuids
    fn.restype = c_int
    result = fn(dev, uuids, entry_length, entry_count)
    _mtmlErrorCheck(result)
    return [uuids[i * entry_length: (i + 1) * entry_length].decode().strip('\x00') for i in range(entry_count)]

def mtmlDeviceCountMaxVirtDevices(dev, virt_type):
    virt_devices_count = c_uint()
    fn = _lib.mtmlDeviceCountMaxVirtDevices
    fn.restype = c_int
    result = fn(dev, virt_type, byref(virt_devices_count))
    _mtmlErrorCheck(result)
    return virt_devices_count.value

def mtmlDeviceInitVirtDevice(dev, uuid):
    virt_dev = POINTER(struct_c_mtmlDevice_t)()
    fn = _lib.mtmlDeviceInitVirtDevice
    fn.restype = c_int
    result = fn(dev, uuid.encode(), byref(virt_dev))
    _mtmlErrorCheck(result)
    return virt_dev

def mtmlDeviceFreeVirtDevice(virt_dev):
    fn = _lib.mtmlDeviceFreeVirtDevice
    fn.restype = c_int
    result = fn(virt_dev)
    _mtmlErrorCheck(result)

def mtmlDeviceGetVirtType(virt_dev):
    virt_type = MtmlVirtType()
    fn = _lib.mtmlDeviceGetVirtType
    fn.restype = c_int
    result = fn(virt_dev, byref(virt_type))
    _mtmlErrorCheck(result)
    return virt_type

def mtmlDeviceGetPhyDeviceUuid(virt_dev):
    uuid = create_string_buffer(MTML_DEVICE_UUID_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetPhyDeviceUuid
    fn.restype = c_int
    result = fn(virt_dev, uuid, MTML_DEVICE_UUID_BUFFER_SIZE)
    _mtmlErrorCheck(result)
    return uuid.value.decode()

## P2P Functions
def mtmlDeviceGetTopologyLevel(dev1, dev2):
    level = MtmlDeviceTopologyLevel(0)
    fn = _lib.mtmlDeviceGetTopologyLevel
    fn.restype = c_int
    result = fn(dev1, dev2, byref(level))
    _mtmlErrorCheck(result)
    return level

def mtmlDeviceCountDeviceByTopologyLevel(dev, level):
    count = c_uint
    fn = _lib.mtmlDeviceCountDeviceByTopologyLevel
    fn.restype = c_int
    result = fn(dev, level, byref(count))
    _mtmlErrorCheck(result)
    return count.value

def mtmlDeviceGetDeviceByTopologyLevel(dev, level):
    count = mtmlDeviceCountDeviceByTopologyLevel(dev, level)
    device_array = (c_mtmlDevice_t * count)()
    fn = _lib.mtmlDeviceGetDeviceByTopologyLevel
    fn.restype = c_int
    result = fn(dev, level, count, device_array)
    _mtmlErrorCheck(result)
    return [device_array[i] for i in range(count)]

def mtmlDeviceGetP2PStatus(dev1, dev2, p2pCap):
    p2pStatus = MtmlDeviceP2PStatus(0) 
    fn = _lib.mtmlDeviceGetP2PStatus
    fn.restype = c_int
    result = fn(dev1, dev2, p2pCap, byref(p2pStatus))
    _mtmlErrorCheck(result)
    return p2pStatus

## GPU Functions
def mtmlGpuGetUtilization(gpu):
    utilization = c_uint()
    fn = _lib.mtmlGpuGetUtilization
    fn.restype = c_int
    result = fn(gpu, byref(utilization))
    _mtmlErrorCheck(result)
    return utilization.value

def mtmlGpuGetTemperature(gpu):
    temp = c_uint()
    fn = _lib.mtmlGpuGetTemperature
    fn.restype = c_int
    result = fn(gpu, byref(temp))
    _mtmlErrorCheck(result)
    return temp.value

def mtmlGpuGetClock(gpu):
    clock_mhz = c_uint()
    fn = _lib.mtmlGpuGetClock
    fn.restype = c_int
    result = fn(gpu, byref(clock_mhz))
    _mtmlErrorCheck(result)
    return clock_mhz.value

def mtmlGpuGetMaxClock(gpu):
    max_clock_mhz = c_uint()
    fn = _lib.mtmlGpuGetMaxClock
    fn.restype = c_int
    result = fn(gpu, byref(max_clock_mhz))
    _mtmlErrorCheck(result)
    return max_clock_mhz.value

def mtmlGpuGetEngineUtilization(gpu, engine):
    utilization = c_uint()
    fn = _lib.mtmlGpuGetEngineUtilization
    fn.restype = c_int
    result = fn(gpu, engine.value, byref(utilization))
    _mtmlErrorCheck(result)
    return utilization.value

## Memory Functions
def mtmlMemoryGetTotal(mem):
    total = c_ulonglong()
    fn = _lib.mtmlMemoryGetTotal
    fn.restype = c_int
    result = fn(mem, byref(total))
    _mtmlErrorCheck(result)
    return total.value

def mtmlMemoryGetUsed(mem):
    used = c_ulonglong()
    fn = _lib.mtmlMemoryGetUsed
    fn.restype = c_int
    result = fn(mem, byref(used))
    _mtmlErrorCheck(result)
    return used.value

def mtmlMemoryGetUtilization(mem):
    utilization = c_uint()
    fn = _lib.mtmlMemoryGetUtilization
    fn.restype = c_int
    result = fn(mem, byref(utilization))
    _mtmlErrorCheck(result)
    return utilization.value

def mtmlMemoryGetClock(mem):
    clock_mhz = c_uint()
    fn = _lib.mtmlMemoryGetClock
    fn.restype = c_int
    result = fn(mem, byref(clock_mhz))
    _mtmlErrorCheck(result)
    return clock_mhz.value

def mtmlMemoryGetMaxClock(mem):
    max_clock_mhz = c_uint()
    fn = _lib.mtmlMemoryGetMaxClock
    fn.restype = c_int
    result = fn(mem, byref(max_clock_mhz))
    _mtmlErrorCheck(result)
    return max_clock_mhz.value

def mtmlMemoryGetBusWidth(mem):
    bus_width = c_uint()
    fn = _lib.mtmlMemoryGetBusWidth
    fn.restype = c_int
    result = fn(mem, byref(bus_width))
    _mtmlErrorCheck(result)
    return bus_width.value

def mtmlMemoryGetBandwidth(mem):
    bandwidth = c_uint()
    fn = _lib.mtmlMemoryGetBandwidth
    fn.restype = c_int
    result = fn(mem, byref(bandwidth))
    _mtmlErrorCheck(result)
    return bandwidth.value

def mtmlMemoryGetSpeed(mem):
    speed = c_uint()
    fn = _lib.mtmlMemoryGetSpeed
    fn.restype = c_int
    result = fn(mem, byref(speed))
    _mtmlErrorCheck(result)
    return speed.value

def mtmlMemoryGetVendor(mem):
    vendor = create_string_buffer(MTML_MEMORY_VENDOR_BUFFER_SIZE)
    fn = _lib.mtmlMemoryGetVendor
    fn.restype = c_int
    result = fn(mem, MTML_MEMORY_VENDOR_BUFFER_SIZE, vendor)
    _mtmlErrorCheck(result)
    return vendor.value.decode()

def mtmlMemoryGetType(mem):
    mem_type = MtmlMemoryType(0)
    fn = _lib.mtmlMemoryGetType
    fn.restype = c_int
    result = fn(mem, byref(mem_type))
    _mtmlErrorCheck(result)
    return mem_type

## VPU Functions
def mtmlVpuGetUtilization(vpu):
    utilization = MtmlCodecUtil()
    fn = _lib.mtmlVpuGetUtilization
    fn.restype = c_int
    result = fn(vpu, byref(utilization))
    _mtmlErrorCheck(result)
    return utilization

def mtmlVpuGetClock(vpu):
    clock_mhz = c_uint()
    fn = _lib.mtmlVpuGetClock
    fn.restype = c_int
    result = fn(vpu, byref(clock_mhz))
    _mtmlErrorCheck(result)
    return clock_mhz.value

def mtmlVpuGetMaxClock(vpu):
    max_clock_mhz = c_uint()
    fn = _lib.mtmlVpuGetMaxClock
    fn.restype = c_int
    result = fn(vpu, byref(max_clock_mhz))
    _mtmlErrorCheck(result)
    return max_clock_mhz.value

def mtmlVpuGetCodecCapacity(vpu):
    encode_capacity = c_uint()
    decode_capacity = c_uint()
    fn = _lib.mtmlVpuGetCodecCapacity
    fn.restype = c_int
    result = fn(vpu, byref(encode_capacity), byref(decode_capacity))
    _mtmlErrorCheck(result)
    return encode_capacity.value, decode_capacity.value

def mtmlVpuGetEncoderSessionStates(vpu, length):
    states = (MtmlCodecSessionState * length)()
    fn = _lib.mtmlVpuGetEncoderSessionStates
    fn.restype = c_int
    result = fn(vpu, states, length)
    _mtmlErrorCheck(result)
    return states

def mtmlVpuGetEncoderSessionMetrics(vpu, session_id):
    metrics = MtmlCodecSessionMetrics()
    fn = _lib.mtmlVpuGetEncoderSessionMetrics
    fn.restype = c_int
    result = fn(vpu, session_id, byref(metrics))
    _mtmlErrorCheck(result)
    return metrics

def mtmlVpuGetDecoderSessionStates(vpu, length):
    states = (MtmlCodecSessionState * length)() 
    fn = _lib.mtmlVpuGetDecoderSessionStates
    fn.restype = c_int
    result = fn(vpu, states, length)
    _mtmlErrorCheck(result)
    return states

def mtmlVpuGetDecoderSessionMetrics(vpu, session_id):
    metrics = MtmlCodecSessionMetrics() 
    fn = _lib.mtmlVpuGetDecoderSessionMetrics
    fn.restype = c_int
    result = fn(vpu, session_id, byref(metrics))
    _mtmlErrorCheck(result)
    return metrics

## Logging Functions
def mtmlLogSetConfiguration(lib, configuration):
    fn = _lib.mtmlLogSetConfiguration
    fn.restype = c_int
    result = fn(byref(configuration))
    _mtmlErrorCheck(result)

def mtmlLogGetConfiguration(lib, configuration):
    fn = _lib.mtmlLogGetConfiguration
    fn.restype = c_int
    result = fn(byref(configuration))
    _mtmlErrorCheck(result)
    return configuration

## Error Reporting
def mtmlErrorString(result_):
    fn = _lib.mtmlErrorString
    fn.restype = c_char_p 
    error_string_ptr = fn(result_)
    error_string = error_string_ptr.decode('utf-8') if error_string_ptr else None
    return error_string

# MPC Functions
def mtmlDeviceSetMpcMode(lib, device, mode):
    fn = lib.mtmlDeviceSetMpcMode
    fn.restype = c_int
    result = fn(byref(device), mode)
    _mtmlErrorCheck(result)

def mtmlDeviceGetMpcMode(lib, device):
    currentMode = MtmlMpcMode()
    fn = lib.mtmlDeviceGetMpcMode
    fn.restype = c_int
    result = fn(byref(device), byref(currentMode))
    _mtmlErrorCheck(result)
    return currentMode.value

def mtmlDeviceCountSupportedMpcProfiles(lib, parentDevice):
    count = c_uint()
    fn = lib.mtmlDeviceCountSupportedMpcProfiles
    fn.restype = c_int
    result = fn(byref(parentDevice), byref(count))
    _mtmlErrorCheck(result)
    return count.value

def mtmlDeviceGetSupportedMpcProfiles(lib, parentDevice, count):
    info = (MtmlMpcProfile * count)()
    fn = lib.mtmlDeviceGetSupportedMpcProfiles
    fn.restype = c_int
    result = fn(byref(parentDevice), count, info)
    _mtmlErrorCheck(result)
    return info

def mtmlDeviceCountSupportedMpcConfigurations(lib, parentDevice):
    count = c_uint()
    fn = lib.mtmlDeviceCountSupportedMpcConfigurations
    fn.restype = c_int
    result = fn(byref(parentDevice), byref(count))
    _mtmlErrorCheck(result)
    return count.value

def mtmlDeviceGetSupportedMpcConfigurations(lib, parentDevice, count):
    info = (MtmlMpcConfiguration * count)()
    fn = lib.mtmlDeviceGetSupportedMpcConfigurations
    fn.restype = c_int
    result = fn(byref(parentDevice), count, info)
    _mtmlErrorCheck(result)
    return info

def mtmlDeviceGetMpcConfiguration(lib, parentDevice):
    config = MtmlMpcConfiguration()
    fn = lib.mtmlDeviceGetMpcConfiguration
    fn.restype = c_int
    result = fn(byref(parentDevice), byref(config))
    _mtmlErrorCheck(result)
    return config

def mtmlDeviceGetMpcConfigurationByName(lib, parentDevice, configName):
    config = MtmlMpcConfiguration()
    fn = lib.mtmlDeviceGetMpcConfigurationByName
    fn.restype = c_int
    result = fn(byref(parentDevice), configName.encode('utf-8'), byref(config))
    _mtmlErrorCheck(result)
    return config

def mtmlDeviceSetMpcConfiguration(lib, parentDevice, id):
    fn = lib.mtmlDeviceSetMpcConfiguration
    fn.restype = c_int
    result = fn(byref(parentDevice), id)
    _mtmlErrorCheck(result)

def mtmlDeviceCountMpcInstancesByProfileId(lib, parentDevice, profileId):
    count = c_uint()
    fn = lib.mtmlDeviceCountMpcInstancesByProfileId
    fn.restype = c_int
    result = fn(byref(parentDevice), profileId, byref(count))
    _mtmlErrorCheck(result)
    return count.value

def mtmlDeviceGetMpcInstancesByProfileId(lib, parentDevice, profileId, count):
    mpcInstance = (MtmlDevice * count)()
    fn = lib.mtmlDeviceGetMpcInstancesByProfileId
    fn.restype = c_int
    result = fn(byref(parentDevice), profileId, count, mpcInstance)
    _mtmlErrorCheck(result)
    return mpcInstance

def mtmlDeviceCountMpcInstances(lib, parentDevice):
    count = c_uint()
    fn = lib.mtmlDeviceCountMpcInstances
    fn.restype = c_int
    result = fn(byref(parentDevice), byref(count))
    _mtmlErrorCheck(result)
    return count.value

def mtmlDeviceGetMpcInstances(lib, parentDevice, count):
    mpcInstance = (MtmlDevice * count)()
    fn = lib.mtmlDeviceGetMpcInstances
    fn.restype = c_int
    result = fn(byref(parentDevice), count, mpcInstance)
    _mtmlErrorCheck(result)
    return mpcInstance

def mtmlDeviceGetMpcInstanceByIndex(lib, parentDevice, index):
    mpcInstance = MtmlDevice()
    fn = lib.mtmlDeviceGetMpcInstanceByIndex
    fn.restype = c_int
    result = fn(byref(parentDevice), index, byref(mpcInstance))
    _mtmlErrorCheck(result)
    return mpcInstance

def mtmlDeviceGetMpcParentDevice(lib, mpcInstance):
    parentDevice = MtmlDevice()
    fn = lib.mtmlDeviceGetMpcParentDevice
    fn.restype = c_int
    result = fn(byref(mpcInstance), byref(parentDevice))
    _mtmlErrorCheck(result)
    return parentDevice

def mtmlDeviceGetMpcProfileInfo(lib, mpcInstance):
    profileInfo = MtmlMpcProfile()
    fn = lib.mtmlDeviceGetMpcProfileInfo
    fn.restype = c_int
    result = fn(byref(mpcInstance), byref(profileInfo))
    _mtmlErrorCheck(result)
    return profileInfo

def mtmlDeviceGetMpcInstanceIndex(lib, mpcInstance):
    index = c_uint()
    fn = lib.mtmlDeviceGetMpcInstanceIndex
    fn.restype = c_int
    result = fn(byref(mpcInstance), byref(index))
    _mtmlErrorCheck(result)
    return index.value

## MtLink Functions
def mtmlDeviceGetMtLinkSpec(device):
    spec = MtmlMtLinkSpec()
    fn = _lib.mtmlDeviceGetMtLinkSpec
    fn.restype = c_int
    result = fn(byref(device), byref(spec))
    _mtmlErrorCheck(result)
    return spec

def mtmlDeviceGetMtLinkState(device, linkId):
    state = MtmlMtLinkState()
    fn = _lib.mtmlDeviceGetMtLinkState
    fn.restype = c_int
    result = fn(byref(device), linkId, byref(state))
    _mtmlErrorCheck(result)
    return state.state

def mtmlDeviceGetMtLinkCapStatus(device, linkId, capability):
    status = MtmlMtLinkCapStatus()
    cap = MtmlMtLinkCap(capability)
    fn = _lib.mtmlDeviceGetMtLinkCapStatus
    fn.restype = c_int
    result = fn(byref(device), linkId, byref(cap), byref(status))
    _mtmlErrorCheck(result)
    return status.status

def mtmlDeviceGetMtLinkRemoteDevice(device, linkId):
    remoteDevice = POINTER(MtmlDevice)()
    fn = _lib.mtmlDeviceGetMtLinkRemoteDevice
    fn.restype = c_int
    result = fn(byref(device), linkId, byref(remoteDevice))
    _mtmlErrorCheck(result)
    return remoteDevice

def mtmlDeviceCountMtLinkShortestPaths(localDevice, remoteDevice):
    pathCount = c_uint()
    pathLength = c_uint()
    fn = _lib.mtmlDeviceCountMtLinkShortestPaths
    fn.restype = c_int
    result = fn(byref(localDevice), byref(remoteDevice), byref(pathCount), byref(pathLength))
    _mtmlErrorCheck(result)
    return pathCount.value, pathLength.value

def mtmlDeviceGetMtLinkShortestPaths(localDevice, remoteDevice, pathCount, pathLength):
    paths = (POINTER(MtmlDevice) * (pathCount * pathLength))()
    fn = _lib.mtmlDeviceGetMtLinkShortestPaths
    fn.restype = c_int
    result = fn(byref(localDevice), byref(remoteDevice), pathCount, pathLength, paths)
    _mtmlErrorCheck(result)
    return paths

def mtmlDeviceCountMtLinkLayouts(localDevice, remoteDevice):
    linkCount = c_uint()
    fn = _lib.mtmlDeviceCountMtLinkLayouts
    fn.restype = c_int
    result = fn(byref(localDevice), byref(remoteDevice), byref(linkCount))
    _mtmlErrorCheck(result)
    return linkCount.value

def mtmlDeviceGetMtLinkLayouts(localDevice, remoteDevice, linkCount):
    layouts = (MtmlMtLinkLayout * linkCount)()
    fn = _lib.mtmlDeviceGetMtLinkLayouts
    fn.restype = c_int
    result = fn(byref(localDevice), byref(remoteDevice), linkCount, layouts)
    _mtmlErrorCheck(result)
    return layouts

## Affinity Functions
def mtmlDeviceGetMemoryAffinityWithinNode(lib, device, nodeSetSize):
    nodeSet = (c_ulong * ((nodeSetSize + 1) // 2))()
    fn = lib.mtmlDeviceGetMemoryAffinityWithinNode
    fn.restype = c_int
    result = fn(byref(device), nodeSetSize, nodeSet)
    _mtmlErrorCheck(result)
    return list(nodeSet[:nodeSetSize])

def mtmlDeviceGetCpuAffinityWithinNode(lib, device, cpuSetSize):
    cpuSet = (c_ulong * ((cpuSetSize + 1) // 2))()
    fn = lib.mtmlDeviceGetCpuAffinityWithinNode
    fn.restype = c_int
    result = fn(byref(device), cpuSetSize, cpuSet)
    _mtmlErrorCheck(result)
    return list(cpuSet[:cpuSetSize])

def mtmlDeviceReset(lib, device):
    fn = lib.mtmlDeviceReset
    fn.restype = c_int
    result = fn(byref(device))
    _mtmlErrorCheck(result)

def mtmlMemorySetEccMode(lib, mem, mode):
    fn = lib.mtmlMemorySetEccMode
    fn.restype = c_int
    result = fn(byref(mem), mode)
    _mtmlErrorCheck(result)

def mtmlMemoryGetEccMode(lib, mem):
    currentMode = MtmlEccMode()
    pendingMode = MtmlEccMode()
    fn = lib.mtmlMemoryGetEccMode
    fn.restype = c_int
    result = fn(byref(mem), byref(currentMode), byref(pendingMode))
    _mtmlErrorCheck(result)
    return currentMode.value, pendingMode.value

def mtmlMemoryGetRetiredPagesCount(lib, mem):
    count = MtmlPageRetirementCount()
    fn = lib.mtmlMemoryGetRetiredPagesCount
    fn.restype = c_int
    result = fn(byref(mem), byref(count))
    _mtmlErrorCheck(result)
    return count.value

def mtmlMemoryGetRetiredPages(lib, mem, cause, count):
    pageRetirements = (MtmlPageRetirement * count)()
    fn = lib.mtmlMemoryGetRetiredPages
    fn.restype = c_int
    result = fn(byref(mem), cause, count, pageRetirements)
    _mtmlErrorCheck(result)
    return pageRetirements[:count]

def mtmlMemoryGetRetiredPagesPendingStatus(lib, mem):
    isPending = MtmlRetiredPagesPendingState()
    fn = lib.mtmlMemoryGetRetiredPagesPendingStatus
    fn.restype = c_int
    result = fn(byref(mem), byref(isPending))
    _mtmlErrorCheck(result)
    return isPending.value

### Encapsulate as an external callable interface
class pymtml:
    MtmlLibrary = None

    @staticmethod
    def mtmlInit():
        lib = c_mtmlLibrary_t
        pymtml.MtmlLibrary = mtmlLibraryInit(lib)

    @staticmethod
    def mtmlShutdown():
        if pymtml.MtmlLibrary is not None:
            mtmlLibraryShutDown(pymtml.MtmlLibrary)
            pymtml.MtmlLibrary = None

    @staticmethod
    def mtmlDeviceGetCount():
        if pymtml.MtmlLibrary is None:
            raise Exception("Library not initialized. Call mtmlInit first.")
        return mtmlLibraryCountDevice(pymtml.MtmlLibrary)

    @staticmethod
    def mtmlDeviceGetHandleByIndex(index):
        if pymtml.MtmlLibrary is None:
            raise Exception("Library not initialized. Call mtmlInit first.")
        dev = c_mtmlDevice_t
        return mtmlLibraryInitDeviceByIndex(pymtml.MtmlLibrary, index, dev)
    
    @staticmethod
    def mtmlDeviceGetName(MtmlDevice):
        return mtmlDeviceGetName(MtmlDevice)
    
    @staticmethod
    def mtmlDeviceGetPowerUsage(MtmlDevice):
        return mtmlDeviceGetPowerUsage(MtmlDevice)

    @staticmethod
    def mtmlDeviceGetMemoryInfo(MtmlDevice):
        mem = c_mtmlMemory_t
        mtmlDeviceInitMemory(MtmlDevice, mem)

        total_memory = mtmlMemoryGetTotal(mem)
        used_memory = mtmlMemoryGetUsed(mem)

        MemoryInfo = namedtuple('MemoryInfo', ['total', 'used'])
        return MemoryInfo(total_memory, used_memory)

    @staticmethod
    def mtmlDeviceGetTemperature(MtmlDevice, sensor_type):
        gpu = c_mtmlGpu_t
        mtmlDeviceInitGpu(MtmlDevice, gpu)
        try:
            return mtmlGpuGetTemperature(gpu)
        finally:
            mtmlDeviceFreeGpu(gpu)

    @staticmethod
    def mtmlDeviceGetUtilizationRates(MtmlDevice):
        gpu = c_mtmlGpu_t
        mtmlDeviceInitGpu(MtmlDevice, gpu)
        try:
            utilization = mtmlGpuGetUtilization(gpu)
            return namedtuple('UtilizationRates', ['gpu', 'memory'])(utilization, mtmlMemoryGetUtilization(c_mtmlMemory_t))
        finally:
            mtmlDeviceFreeGpu(gpu)
        
###test###
# if __name__ == "__main__":
#     lib = c_mtmlLibrary_t
#     MtmlLibrary = mtmlLibraryInit(lib)

#     # system 
#     sys = c_mtmlSystem_t
#     MtmlSystem = mtmlLibraryInitSystem(MtmlLibrary, sys)

#     version = mtmlSystemGetDriverVersion(MtmlSystem)
#     print("Driver Version:", version)

#     mtmlLibraryFreeSystem(MtmlSystem)

#     # device
#     dev_count = mtmlLibraryCountDevice(MtmlLibrary)

#     for dev_index in range(dev_count):
#         dev = c_mtmlDevice_t
#         MtmlDevice = mtmlLibraryInitDeviceByIndex(MtmlLibrary, dev_index, dev)

#         try:
#             dev_uuid = mtmlDeviceGetUUID(MtmlDevice)
#             print(f"Device {dev_index} UUID:", dev_uuid)

#             device_name = mtmlDeviceGetName(MtmlDevice)
#             print("Device Name:", device_name)

#             property = mtmlDeviceGetProperty(MtmlDevice)
#             print("Device Property:", property)

#             gpu = c_mtmlGpu_t
#             mtmlDeviceInitGpu(MtmlDevice, gpu)

#             try:
#                 utilization = mtmlGpuGetUtilization(gpu)
#                 print("GPU Utilization:", utilization, "%")

#                 temp = mtmlGpuGetTemperature(gpu)
#                 print("GPU Temperature:", temp, "°C")

#                 clock = mtmlGpuGetClock(gpu)
#                 print("GPU Clock:", clock, "MHz")

#                 max_clock = mtmlGpuGetMaxClock(gpu)
#                 print("GPU Max Clock:", max_clock, "MHz")

#                 # engine_utilization = mtmlGpuGetEngineUtilization(gpu, MtmlGpuEngine.TWO_D)
#                 # print("2D Engine Utilization:", engine_utilization, "%")

#             finally:
#                 mtmlDeviceFreeGpu(gpu)

#             mem = c_mtmlMemory_t
#             mtmlDeviceInitMemory(MtmlDevice, mem)

#             try:
#                 total_memory = mtmlMemoryGetTotal(mem)
#                 print("Total Memory:", total_memory, "Bytes")

#                 used_memory = mtmlMemoryGetUsed(mem)
#                 print("Used Memory:", used_memory, "Bytes")

#                 memory_utilization = mtmlMemoryGetUtilization(mem)
#                 print("Memory Utilization:", memory_utilization, "%")

#                 memory_clock = mtmlMemoryGetClock(mem)
#                 print("Memory Clock:", memory_clock, "MHz")

#                 max_memory_clock = mtmlMemoryGetMaxClock(mem)
#                 print("Max Memory Clock:", max_memory_clock, "MHz")

#                 memory_bus_width = mtmlMemoryGetBusWidth(mem)
#                 print("Memory Bus Width:", memory_bus_width, "bits")

#                 memory_bandwidth = mtmlMemoryGetBandwidth(mem)
#                 print("Memory Bandwidth:", memory_bandwidth, "GB/s")

#             finally:
#                 mtmlDeviceFreeMemory(mem)

#         finally:
#             mtmlLibraryFreeDevice(MtmlDevice)

#     mtmlLibraryShutDown(MtmlLibrary)

if __name__ == "__main__":
    lib = c_mtmlLibrary_t
    MtmlLibrary = mtmlLibraryInit(lib)

    # system 
    sys = c_mtmlSystem_t
    MtmlSystem = mtmlLibraryInitSystem(MtmlLibrary, sys)

    version = mtmlSystemGetDriverVersion(MtmlSystem)
    print("Driver Version:", version)

    mtmlLibraryFreeSystem(MtmlSystem)

    # device
    dev_count = mtmlLibraryCountDevice(MtmlLibrary)

    for dev_index in range(dev_count):
        dev = c_mtmlDevice_t
        MtmlDevice = mtmlLibraryInitDeviceByIndex(MtmlLibrary, dev_index, dev)

        try:
            dev_uuid = mtmlDeviceGetUUID(MtmlDevice)
            print(f"Device {dev_index} UUID:", dev_uuid)

            device_name = mtmlDeviceGetName(MtmlDevice)
            print("Device Name:", device_name)

            property = mtmlDeviceGetProperty(MtmlDevice)
            print("Device Property:", property)

            # GPU
            gpu = c_mtmlGpu_t
            mtmlDeviceInitGpu(MtmlDevice, gpu)

            try:
                utilization = mtmlGpuGetUtilization(gpu)
                print("GPU Utilization:", utilization, "%")

                temp = mtmlGpuGetTemperature(gpu)
                print("GPU Temperature:", temp, "°C")

                clock = mtmlGpuGetClock(gpu)
                print("GPU Clock:", clock, "MHz")

                max_clock = mtmlGpuGetMaxClock(gpu)
                print("GPU Max Clock:", max_clock, "MHz")

                # engine_utilization = mtmlGpuGetEngineUtilization(gpu, MtmlGpuEngine.TWO_D)
                # print("2D Engine Utilization:", engine_utilization, "%")

            finally:
                mtmlDeviceFreeGpu(gpu)

            # Memory
            mem = c_mtmlMemory_t
            mtmlDeviceInitMemory(MtmlDevice, mem)

            try:
                total_memory = mtmlMemoryGetTotal(mem)
                print("Total Memory:", total_memory, "Bytes")

                used_memory = mtmlMemoryGetUsed(mem)
                print("Used Memory:", used_memory, "Bytes")

                memory_utilization = mtmlMemoryGetUtilization(mem)
                print("Memory Utilization:", memory_utilization, "%")

                memory_clock = mtmlMemoryGetClock(mem)
                print("Memory Clock:", memory_clock, "MHz")

                max_memory_clock = mtmlMemoryGetMaxClock(mem)
                print("Max Memory Clock:", max_memory_clock, "MHz")

                memory_bus_width = mtmlMemoryGetBusWidth(mem)
                print("Memory Bus Width:", memory_bus_width, "bits")

                memory_bandwidth = mtmlMemoryGetBandwidth(mem)
                print("Memory Bandwidth:", memory_bandwidth, "GB/s")

            finally:
                mtmlDeviceFreeMemory(mem)

            # VPU
            vpu = c_mtmlVpu_t
            mtmlDeviceInitVpu(MtmlDevice, vpu)

            try:
                utilization = mtmlVpuGetUtilization(vpu)
                print("VPU Utilization:", utilization)

                clock = mtmlVpuGetClock(vpu)
                print("VPU Clock:", clock, "MHz")

                max_clock = mtmlVpuGetMaxClock(vpu)
                print("VPU Max Clock:", max_clock, "MHz")

                encode_capacity, decode_capacity = mtmlVpuGetCodecCapacity(vpu)
                print("VPU Encode Capacity:", encode_capacity)
                print("VPU Decode Capacity:", decode_capacity)

            finally:
                mtmlDeviceFreeVpu(vpu)

            # Fans
            fan_count = mtmlDeviceCountFan(MtmlDevice)
            print("Fan Count:", fan_count)

            for fan_index in range(fan_count):
                speed = mtmlDeviceGetFanSpeed(MtmlDevice, fan_index)
                print(f"Fan {fan_index} Speed:", speed, "RPM")

            pci_info = mtmlDeviceGetPciInfo(MtmlDevice)
            print("PCI Info:", pci_info)

            power_usage = mtmlDeviceGetPowerUsage(MtmlDevice)
            print("Power Usage:", power_usage, "W")

            gpu_path = mtmlDeviceGetGpuPath(MtmlDevice)
            print("GPU Path:", gpu_path)

            primary_path = mtmlDeviceGetPrimaryPath(MtmlDevice)
            print("Primary Path:", primary_path)

            render_path = mtmlDeviceGetRenderPath(MtmlDevice)
            print("Render Path:", render_path)

            mtbios_version = mtmlDeviceGetMtBiosVersion(MtmlDevice)
            print("MT BIOS Version:", mtbios_version)

            serial_number = mtmlDeviceGetSerialNumber(MtmlDevice)
            print("Serial Number:", serial_number)

            virt_types = mtmlDeviceGetSupportedVirtTypes(MtmlDevice)
            print("Supported Virt Types:", virt_types)

            active_virt_dev_count = mtmlDeviceCountActiveVirtDevices(MtmlDevice)
            print("Active Virt Device Count:", active_virt_dev_count)

            for virt_index in range(active_virt_dev_count):
                uuid = mtmlDeviceGetActiveVirtDeviceUuids(MtmlDevice, MTML_DEVICE_UUID_BUFFER_SIZE, virt_index)
                print(f"Active Virt Device {virt_index} UUID:", uuid)

        finally:
            mtmlLibraryFreeDevice(MtmlDevice)

    mtmlLibraryShutDown(MtmlLibrary)