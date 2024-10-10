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
class MtmlGpuEngine(Enum):
    GEOMETRY = 0
    TWO_D = 1
    THREE_D = 2
    COMPUTE = 3
    MAX = 4

class MtmlMemoryType(Enum):
    MTML_MEM_TYPE_LPDDR4 = 0
    MTML_MEM_TYPE_GDDR6 = 1
    
class MtmlDeviceTopologyLevel(Enum):
    MTML_TOPOLOGY_INTERNAL = 0
    MTML_TOPOLOGY_SINGLE = 1
    MTML_TOPOLOGY_MULTIPLE = 2
    MTML_TOPOLOGY_HOSTBRIDGE = 3
    MTML_TOPOLOGY_NODE = 4
    MTML_TOPOLOGY_SYSTEM = 5

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

class MtmlDeviceP2PStatus(Enum):
    MTML_P2P_STATUS_OK = 0
    MTML_P2P_STATUS_CHIPSET_NOT_SUPPORTED = 1
    MTML_P2P_STATUS_GPU_NOT_SUPPORTED = 2
    MTML_P2P_STATUS_UNKNOWN = 3
    
class MtmlBrandType(Enum):
    MTML_BRAND_MTT = 0
    MTML_BRAND_UNKNOWN = 1
    MTML_BRAND_COUNT = 2
    
class MtmlCodecSessionState(Enum):
    MTML_CODEC_SESSION_STATE_UNKNOWN = -1
    MTML_CODEC_SESSION_STATE_IDLE = 0
    MTML_CODEC_SESSION_STATE_ACTIVE = 1
    MTML_CODEC_SESSION_STATE_COUNT = 2

### Load MTML
_lib = ctypes.CDLL("libmtml.so")

### Error Process
# def _mtmlCheckReturn(result):
#     if result != MtmlReturn.MTML_SUCCESS.value:
#         raise Exception("MTML Error Code: {}".format(result))
class MTMLError(Exception):
    _errcode_to_string = {
        MtmlReturn.MTML_SUCCESS: "Success",
        MtmlReturn.MTML_ERROR_DRIVER_NOT_LOADED: "Driver Not Loaded",
        MtmlReturn.MTML_ERROR_DRIVER_FAILURE: "Driver Failure",
        MtmlReturn.MTML_ERROR_INVALID_ARGUMENT: "Invalid Argument",
        MtmlReturn.MTML_ERROR_NOT_SUPPORTED: "Not Supported",
        MtmlReturn.MTML_ERROR_NO_PERMISSION: "Insufficient Permissions",
        MtmlReturn.MTML_ERROR_INSUFFICIENT_SIZE: "Insufficient Size",
        MtmlReturn.MTML_ERROR_NOT_FOUND: "Not Found",
        MtmlReturn.MTML_ERROR_INSUFFICIENT_MEMORY: "Insufficient Memory",
        MtmlReturn.MTML_ERROR_DRIVER_TOO_OLD: "Driver Too Old",
        MtmlReturn.MTML_ERROR_DRIVER_TOO_NEW: "Driver Too New",
        MtmlReturn.MTML_ERROR_TIMEOUT: "Timeout",
        MtmlReturn.MTML_ERROR_UNKNOWN: "Unknown Error",
    }

    def __new__(cls, value):
        if not isinstance(value, MtmlReturn):
            raise TypeError("Expected an MtmlReturn enum member")
        if value not in cls._errcode_to_string:
            raise ValueError("Unknown error code")
        return super(MTMLError, cls).__new__(cls)

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self._errcode_to_string[self.value]

def _mtmlCheckReturn(ret):
    if ret != MtmlReturn.MTML_SUCCESS.value:
        raise MTMLError(MtmlReturn(ret))

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

class MtmlDispIntfSpec(_PrintableStructure):
    _fields_ = [
        ("type", c_uint),                 # Display interface type
        ("maxHoriRes", c_uint),          # Maximum horizontal resolution
        ("maxVertRes", c_uint),          # Maximum vertical resolution
        ("maxRefreshRate", c_float),     # Maximum refresh rate
        ("rsvd", c_uint * 8)             # Reserved field
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

## MtmlLogConfiguration 
class MtmlLogLevel(Enum):
    MTML_LOG_LEVEL_DEBUG = 0
    MTML_LOG_LEVEL_INFO = 1
    MTML_LOG_LEVEL_WARN = 2
    MTML_LOG_LEVEL_ERROR = 3
    MTML_LOG_LEVEL_FATAL = 4

class MtmlConsoleConfig(Structure):
    _fields_ = [
        ("level", c_uint),  # Log level
        ("rsvd", c_int * 2) # Reserved field
    ]

class MtmlSystemConfig(Structure):
    _fields_ = [
        ("level", c_uint),  # Log level
        ("rsvd", c_int * 2) # Reserved field
    ]

class MtmlFileConfig(Structure):
    _fields_ = [
        ("level", c_uint),  # Log level
        ("file", c_char * 200),  # Log file path
        ("size", c_uint),  # Maximum log file size
        ("rsvd", c_int * 2) # Reserved field
    ]

class MtmlCallbackConfig(Structure):
    _fields_ = [
        ("level", c_uint),  # Log level
        ("callback", CFUNCTYPE(None, c_char_p, c_uint)),  # Callback function
        ("rsvd", c_int * 2) # Reserved field
    ]

class MtmlLogConfiguration(Structure):
    _fields_ = [
        ("consoleConfig", MtmlConsoleConfig),
        ("systemConfig", MtmlSystemConfig),
        ("fileConfig", MtmlFileConfig),
        ("callbackConfig", MtmlCallbackConfig)
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
    _mtmlCheckReturn(result)
    return lib

def mtmlLibraryShutDown(lib):
    fn = _lib.mtmlLibraryShutDown
    fn.restype = c_int
    result = fn(lib)
    _mtmlCheckReturn(result)

def mtmlLibraryGetVersion(lib):
    version = create_string_buffer(MTML_LIBRARY_VERSION_BUFFER_SIZE)
    length = c_uint(MTML_LIBRARY_VERSION_BUFFER_SIZE)
    fn = _lib.mtmlLibraryGetVersion
    fn.restype = c_int
    result = fn(lib, version, length)
    _mtmlCheckReturn(result)
    return version.value.decode()

def mtmlLibraryInitSystem(lib, sys):
    fn = _lib.mtmlLibraryInitSystem
    fn.restype = c_int
    result = fn(lib, byref(sys))
    _mtmlCheckReturn(result)
    return sys

def mtmlLibraryFreeSystem(sys):
    fn = _lib.mtmlLibraryFreeSystem
    fn.restype = c_int
    result = fn(sys)
    _mtmlCheckReturn(result)

def mtmlLibraryCountDevice(lib):
    count = c_uint()
    fn = _lib.mtmlLibraryCountDevice
    fn.restype = c_int
    result = fn(lib, byref(count))
    _mtmlCheckReturn(result)
    return count.value

def mtmlLibraryInitDeviceByIndex(lib, index, dev):
    fn = _lib.mtmlLibraryInitDeviceByIndex
    fn.restype = c_int
    result = fn(lib, index, byref(dev))
    _mtmlCheckReturn(result)
    return dev

def mtmlLibraryInitDeviceByUuid(lib, uuid, dev):
    fn = _lib.mtmlLibraryInitDeviceByUuid
    fn.restype = c_int
    result = fn(lib, uuid.encode(), byref(dev))
    _mtmlCheckReturn(result)
    return dev

def mtmlLibraryInitDeviceByPciSbdf(lib, pciSbdf, dev):
    fn = _lib.mtmlLibraryInitDeviceByPciSbdf
    fn.restype = c_int
    result = fn(lib, pciSbdf.encode(), byref(dev))
    _mtmlCheckReturn(result)
    return dev

def mtmlLibrarySetMpcConfigurationInBatch(lib, count, devices, mpcConfigIds):
    fn = _lib.mtmlLibrarySetMpcConfigurationInBatch
    fn.restype = c_int
    result = fn(lib, count, devices, mpcConfigIds)
    _mtmlCheckReturn(result)

def mtmlLibraryFreeDevice(dev):
    fn = _lib.mtmlLibraryFreeDevice
    fn.restype = c_int
    result = fn(dev)
    _mtmlCheckReturn(result)

## System Functions
def mtmlSystemGetDriverVersion(sys):
    version = create_string_buffer(MTML_DRIVER_VERSION_BUFFER_SIZE)
    length = c_uint(MTML_DRIVER_VERSION_BUFFER_SIZE)
    fn = _lib.mtmlSystemGetDriverVersion
    fn.restype = c_int
    result = fn(sys, version, length)
    _mtmlCheckReturn(result)
    return version.value.decode()

## Device Functions
def mtmlDeviceInitGpu(dev, gpu):
    fn = _lib.mtmlDeviceInitGpu
    fn.restype = c_int
    result = fn(dev, byref(gpu))
    _mtmlCheckReturn(result)
    return gpu

def mtmlDeviceFreeGpu(gpu):
    fn = _lib.mtmlDeviceFreeGpu
    fn.restype = c_int
    result = fn(gpu)
    _mtmlCheckReturn(result)

def mtmlDeviceInitMemory(dev, mem):
    fn = _lib.mtmlDeviceInitMemory
    fn.restype = c_int
    result = fn(dev, byref(mem))
    _mtmlCheckReturn(result)
    return mem

def mtmlDeviceFreeMemory(mem):
    fn = _lib.mtmlDeviceFreeMemory
    fn.restype = c_int
    result = fn(mem)
    _mtmlCheckReturn(result)

def mtmlDeviceInitVpu(dev, vpu):
    fn = _lib.mtmlDeviceInitVpu
    fn.restype = c_int
    result = fn(dev, byref(vpu))
    _mtmlCheckReturn(result)
    return vpu

def mtmlDeviceFreeVpu(vpu):
    fn = _lib.mtmlDeviceFreeVpu
    fn.restype = c_int
    result = fn(vpu)
    _mtmlCheckReturn(result)

def mtmlDeviceGetIndex(dev):
    index = c_uint()
    fn = _lib.mtmlDeviceGetIndex
    fn.restype = c_int
    result = fn(dev, byref(index))
    _mtmlCheckReturn(result)
    return index.value

def mtmlDeviceGetUUID(dev):
    uuid = create_string_buffer(MTML_DEVICE_UUID_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_UUID_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetUUID
    fn.restype = c_int
    result = fn(dev, uuid, length)
    _mtmlCheckReturn(result)
    return uuid.value.decode()

def mtmlDeviceGetBrand(dev):
    brand_type = c_int()
    fn = _lib.mtmlDeviceGetBrand
    fn.restype = c_int
    result = fn(dev, byref(brand_type))
    _mtmlCheckReturn(result)
    return MtmlBrandType(brand_type.value)

def mtmlDeviceGetName(dev):
    name = create_string_buffer(MTML_DEVICE_NAME_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_NAME_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetName
    fn.restype = c_int
    result = fn(dev, name, length)
    _mtmlCheckReturn(result)
    return name.value.decode()

def mtmlDeviceGetPciInfo(dev):
    pci_info = MtmlPciInfo()
    fn = _lib.mtmlDeviceGetPciInfo
    fn.restype = c_int
    result = fn(dev, byref(pci_info))
    _mtmlCheckReturn(result)
    return pci_info

def mtmlDeviceGetPowerUsage(dev):
    power = c_uint()
    fn = _lib.mtmlDeviceGetPowerUsage
    fn.restype = c_int
    result = fn(dev, byref(power))
    _mtmlCheckReturn(result)
    return power.value

def mtmlDeviceGetGpuPath(dev):
    path = create_string_buffer(MTML_DEVICE_PATH_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_PATH_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetGpuPath
    fn.restype = c_int
    result = fn(dev, path, length)
    _mtmlCheckReturn(result)
    return path.value.decode()

def mtmlDeviceGetPrimaryPath(dev):
    path = create_string_buffer(MTML_DEVICE_PATH_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_PATH_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetPrimaryPath
    fn.restype = c_int
    result = fn(dev, path, length)
    _mtmlCheckReturn(result)
    return path.value.decode()

def mtmlDeviceGetRenderPath(dev):
    path = create_string_buffer(MTML_DEVICE_PATH_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_PATH_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetRenderPath
    fn.restype = c_int
    result = fn(dev, path, length)
    _mtmlCheckReturn(result)
    return path.value.decode()

def mtmlDeviceGetMtBiosVersion(dev):
    version = create_string_buffer(MTML_DEVICE_MTBIOS_VERSION_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_MTBIOS_VERSION_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetMtBiosVersion
    fn.restype = c_int
    result = fn(dev, version, length)
    _mtmlCheckReturn(result)
    return version.value.decode()

def mtmlDeviceGetProperty(dev):
    prop = MtmlDeviceProperty()
    fn = _lib.mtmlDeviceGetProperty
    fn.restype = c_int
    result = fn(dev, byref(prop))
    _mtmlCheckReturn(result)
    return prop

def mtmlDeviceCountFan(dev):
    count = c_uint()
    fn = _lib.mtmlDeviceCountFan
    fn.restype = c_int
    result = fn(dev, byref(count))
    _mtmlCheckReturn(result)
    return count.value

def mtmlDeviceGetFanSpeed(dev, index):
    speed = c_uint()
    fn = _lib.mtmlDeviceGetFanSpeed
    fn.restype = c_int
    result = fn(dev, index, byref(speed))
    _mtmlCheckReturn(result)
    return speed.value

def mtmlDeviceGetPcieSlotInfo(dev):
    slot_info = MtmlPciSlotInfo()
    fn = _lib.mtmlDeviceGetPcieSlotInfo
    fn.restype = c_int
    result = fn(dev, byref(slot_info))
    _mtmlCheckReturn(result)
    return slot_info

def mtmlDeviceCountDisplayInterface(dev):
    count = c_uint()
    fn = _lib.mtmlDeviceCountDisplayInterface
    fn.restype = c_int
    result = fn(dev, byref(count))
    _mtmlCheckReturn(result)
    return count.value

def mtmlDeviceGetDisplayInterfaceSpec(dev, intfIndex):
    spec = MtmlDispIntfSpec()
    fn = _lib.mtmlDeviceGetDisplayInterfaceSpec
    fn.restype = c_int
    result = fn(dev, intfIndex, byref(spec))
    _mtmlCheckReturn(result)
    return spec

def mtmlDeviceGetSerialNumber(dev):
    serial_number = create_string_buffer(MTML_DEVICE_SERIAL_NUMBER_BUFFER_SIZE)
    length = c_int(MTML_DEVICE_SERIAL_NUMBER_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetSerialNumber
    fn.restype = c_int
    result = fn(dev, length, serial_number)
    _mtmlCheckReturn(result)
    return serial_number.value.decode()

## Device Virtualization Functions
def mtmlDeviceCountSupportedVirtTypes(dev):
    count = c_uint()
    fn = _lib.mtmlDeviceCountSupportedVirtTypes
    fn.restype = c_int
    result = fn(dev, byref(count))
    _mtmlCheckReturn(result)
    return count.value

def mtmlDeviceGetSupportedVirtTypes(dev):
    types = (MtmlVirtType * 10)()
    fn = _lib.mtmlDeviceGetSupportedVirtTypes
    fn.restype = c_int
    result = fn(dev, types, len(types))
    _mtmlCheckReturn(result)
    return [types[i] for i in range(len(types))]

def mtmlDeviceCountAvailVirtTypes(dev):
    count = c_uint()
    fn = _lib.mtmlDeviceCountAvailVirtTypes
    fn.restype = c_int
    result = fn(dev, byref(count))
    _mtmlCheckReturn(result)
    return count.value

def mtmlDeviceGetAvailVirtTypes(dev):
    types = (MtmlVirtType * 10)()
    fn = _lib.mtmlDeviceGetAvailVirtTypes
    fn.restype = c_int
    result = fn(dev, types, len(types))
    _mtmlCheckReturn(result)
    return [types[i] for i in range(len(types))]

def mtmlDeviceCountAvailVirtDevices(dev, virt_type):
    count = c_uint()
    fn = _lib.mtmlDeviceCountAvailVirtDevices
    fn.restype = c_int
    result = fn(dev, virt_type, byref(count))
    _mtmlCheckReturn(result)
    return count.value

def mtmlDeviceCountActiveVirtDevices(dev):
    count = c_uint()
    fn = _lib.mtmlDeviceCountActiveVirtDevices
    fn.restype = c_int
    result = fn(dev, byref(count))
    _mtmlCheckReturn(result)
    return count.value

def mtmlDeviceGetActiveVirtDeviceUuids(dev, entry_length, entry_count):
    uuids = create_string_buffer(entry_length * entry_count)
    fn = _lib.mtmlDeviceGetActiveVirtDeviceUuids
    fn.restype = c_int
    result = fn(dev, uuids, entry_length, entry_count)
    _mtmlCheckReturn(result)
    return [uuids[i * entry_length: (i + 1) * entry_length].decode().strip('\x00') for i in range(entry_count)]

def mtmlDeviceCountMaxVirtDevices(dev, virt_type):
    virt_devices_count = c_uint()
    fn = _lib.mtmlDeviceCountMaxVirtDevices
    fn.restype = c_int
    result = fn(dev, virt_type, byref(virt_devices_count))
    _mtmlCheckReturn(result)
    return virt_devices_count.value

def mtmlDeviceInitVirtDevice(dev, uuid):
    virt_dev = POINTER(struct_c_mtmlDevice_t)()
    fn = _lib.mtmlDeviceInitVirtDevice
    fn.restype = c_int
    result = fn(dev, uuid.encode(), byref(virt_dev))
    _mtmlCheckReturn(result)
    return virt_dev

def mtmlDeviceFreeVirtDevice(virt_dev):
    fn = _lib.mtmlDeviceFreeVirtDevice
    fn.restype = c_int
    result = fn(virt_dev)
    _mtmlCheckReturn(result)

def mtmlDeviceGetVirtType(virt_dev):
    virt_type = MtmlVirtType()
    fn = _lib.mtmlDeviceGetVirtType
    fn.restype = c_int
    result = fn(virt_dev, byref(virt_type))
    _mtmlCheckReturn(result)
    return virt_type

def mtmlDeviceGetPhyDeviceUuid(virt_dev):
    uuid = create_string_buffer(MTML_DEVICE_UUID_BUFFER_SIZE)
    fn = _lib.mtmlDeviceGetPhyDeviceUuid
    fn.restype = c_int
    result = fn(virt_dev, uuid, MTML_DEVICE_UUID_BUFFER_SIZE)
    _mtmlCheckReturn(result)
    return uuid.value.decode()

## P2P Functions
def mtmlDeviceGetTopologyLevel(dev1, dev2):
    level = MtmlDeviceTopologyLevel(0)
    fn = _lib.mtmlDeviceGetTopologyLevel
    fn.restype = c_int
    result = fn(dev1, dev2, byref(level))
    _mtmlCheckReturn(result)
    return level

def mtmlDeviceCountDeviceByTopologyLevel(dev, level):
    count = c_uint()
    fn = _lib.mtmlDeviceCountDeviceByTopologyLevel
    fn.restype = c_int
    result = fn(dev, level, byref(count))
    _mtmlCheckReturn(result)
    return count.value

def mtmlDeviceGetDeviceByTopologyLevel(dev, level):
    count = mtmlDeviceCountDeviceByTopologyLevel(dev, level)
    device_array = (c_mtmlDevice_t * count)()
    fn = _lib.mtmlDeviceGetDeviceByTopologyLevel
    fn.restype = c_int
    result = fn(dev, level, count, device_array)
    _mtmlCheckReturn(result)
    return [device_array[i] for i in range(count)]

def mtmlDeviceGetP2PStatus(dev1, dev2, p2pCap):
    p2pStatus = MtmlDeviceP2PStatus(0) 
    fn = _lib.mtmlDeviceGetP2PStatus
    fn.restype = c_int
    result = fn(dev1, dev2, p2pCap, byref(p2pStatus))
    _mtmlCheckReturn(result)
    return p2pStatus

## GPU Functions
def mtmlGpuGetUtilization(gpu):
    utilization = c_uint()
    fn = _lib.mtmlGpuGetUtilization
    fn.restype = c_int
    result = fn(gpu, byref(utilization))
    _mtmlCheckReturn(result)
    return utilization.value

def mtmlGpuGetTemperature(gpu):
    temp = c_uint()
    fn = _lib.mtmlGpuGetTemperature
    fn.restype = c_int
    result = fn(gpu, byref(temp))
    _mtmlCheckReturn(result)
    return temp.value

def mtmlGpuGetClock(gpu):
    clock_mhz = c_uint()
    fn = _lib.mtmlGpuGetClock
    fn.restype = c_int
    result = fn(gpu, byref(clock_mhz))
    _mtmlCheckReturn(result)
    return clock_mhz.value

def mtmlGpuGetMaxClock(gpu):
    max_clock_mhz = c_uint()
    fn = _lib.mtmlGpuGetMaxClock
    fn.restype = c_int
    result = fn(gpu, byref(max_clock_mhz))
    _mtmlCheckReturn(result)
    return max_clock_mhz.value

def mtmlGpuGetEngineUtilization(gpu, engine):
    utilization = c_uint()
    fn = _lib.mtmlGpuGetEngineUtilization
    fn.restype = c_int
    result = fn(gpu, engine.value, byref(utilization))
    _mtmlCheckReturn(result)
    return utilization.value

## Memory Functions
def mtmlMemoryGetTotal(mem):
    total = c_ulonglong()
    fn = _lib.mtmlMemoryGetTotal
    fn.restype = c_int
    result = fn(mem, byref(total))
    _mtmlCheckReturn(result)
    return total.value

def mtmlMemoryGetUsed(mem):
    used = c_ulonglong()
    fn = _lib.mtmlMemoryGetUsed
    fn.restype = c_int
    result = fn(mem, byref(used))
    _mtmlCheckReturn(result)
    return used.value

def mtmlMemoryGetUtilization(mem):
    utilization = c_uint()
    fn = _lib.mtmlMemoryGetUtilization
    fn.restype = c_int
    result = fn(mem, byref(utilization))
    _mtmlCheckReturn(result)
    return utilization.value

def mtmlMemoryGetClock(mem):
    clock_mhz = c_uint()
    fn = _lib.mtmlMemoryGetClock
    fn.restype = c_int
    result = fn(mem, byref(clock_mhz))
    _mtmlCheckReturn(result)
    return clock_mhz.value

def mtmlMemoryGetMaxClock(mem):
    max_clock_mhz = c_uint()
    fn = _lib.mtmlMemoryGetMaxClock
    fn.restype = c_int
    result = fn(mem, byref(max_clock_mhz))
    _mtmlCheckReturn(result)
    return max_clock_mhz.value

def mtmlMemoryGetBusWidth(mem):
    bus_width = c_uint()
    fn = _lib.mtmlMemoryGetBusWidth
    fn.restype = c_int
    result = fn(mem, byref(bus_width))
    _mtmlCheckReturn(result)
    return bus_width.value

def mtmlMemoryGetBandwidth(mem):
    bandwidth = c_uint()
    fn = _lib.mtmlMemoryGetBandwidth
    fn.restype = c_int
    result = fn(mem, byref(bandwidth))
    _mtmlCheckReturn(result)
    return bandwidth.value

def mtmlMemoryGetSpeed(mem):
    speed = c_uint()
    fn = _lib.mtmlMemoryGetSpeed
    fn.restype = c_int
    result = fn(mem, byref(speed))
    _mtmlCheckReturn(result)
    return speed.value

def mtmlMemoryGetVendor(mem):
    vendor = create_string_buffer(MTML_MEMORY_VENDOR_BUFFER_SIZE)
    fn = _lib.mtmlMemoryGetVendor
    fn.restype = c_int
    result = fn(mem, MTML_MEMORY_VENDOR_BUFFER_SIZE, vendor)
    _mtmlCheckReturn(result)
    return vendor.value.decode()

def mtmlMemoryGetType(mem):
    mem_type = MtmlMemoryType(0)
    fn = _lib.mtmlMemoryGetType
    fn.restype = c_int
    result = fn(mem, byref(mem_type))
    _mtmlCheckReturn(result)
    return mem_type

## VPU Functions
def mtmlVpuGetUtilization(vpu):
    utilization = MtmlCodecUtil()
    fn = _lib.mtmlVpuGetUtilization
    fn.restype = c_int
    result = fn(vpu, byref(utilization))
    _mtmlCheckReturn(result)
    return utilization

def mtmlVpuGetClock(vpu):
    clock_mhz = c_uint()
    fn = _lib.mtmlVpuGetClock
    fn.restype = c_int
    result = fn(vpu, byref(clock_mhz))
    _mtmlCheckReturn(result)
    return clock_mhz.value

def mtmlVpuGetMaxClock(vpu):
    max_clock_mhz = c_uint()
    fn = _lib.mtmlVpuGetMaxClock
    fn.restype = c_int
    result = fn(vpu, byref(max_clock_mhz))
    _mtmlCheckReturn(result)
    return max_clock_mhz.value

def mtmlVpuGetCodecCapacity(vpu):
    encode_capacity = c_uint()
    decode_capacity = c_uint()
    fn = _lib.mtmlVpuGetCodecCapacity
    fn.restype = c_int
    result = fn(vpu, byref(encode_capacity), byref(decode_capacity))
    _mtmlCheckReturn(result)
    return encode_capacity.value, decode_capacity.value

def mtmlVpuGetEncoderSessionStates(vpu, length):
    states = (MtmlCodecSessionState * length)()
    fn = _lib.mtmlVpuGetEncoderSessionStates
    fn.restype = c_int
    result = fn(vpu, states, length)
    _mtmlCheckReturn(result)
    return states

def mtmlVpuGetEncoderSessionMetrics(vpu, session_id):
    metrics = MtmlCodecSessionMetrics()
    fn = _lib.mtmlVpuGetEncoderSessionMetrics
    fn.restype = c_int
    result = fn(vpu, session_id, byref(metrics))
    _mtmlCheckReturn(result)
    return metrics

def mtmlVpuGetDecoderSessionStates(vpu, length):
    states = (MtmlCodecSessionState * length)() 
    fn = _lib.mtmlVpuGetDecoderSessionStates
    fn.restype = c_int
    result = fn(vpu, states, length)
    _mtmlCheckReturn(result)
    return states

def mtmlVpuGetDecoderSessionMetrics(vpu, session_id):
    metrics = MtmlCodecSessionMetrics() 
    fn = _lib.mtmlVpuGetDecoderSessionMetrics
    fn.restype = c_int
    result = fn(vpu, session_id, byref(metrics))
    _mtmlCheckReturn(result)
    return metrics

## Logging Functions
def mtmlLogSetConfiguration(lib, configuration):
    fn = _lib.mtmlLogSetConfiguration
    fn.restype = c_int
    result = fn(byref(configuration))
    _mtmlCheckReturn(result)

def mtmlLogGetConfiguration(lib, configuration):
    fn = _lib.mtmlLogGetConfiguration
    fn.restype = c_int
    result = fn(byref(configuration))
    _mtmlCheckReturn(result)
    return configuration

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
        

        







if __name__ == "__main__":
    lib = c_mtmlLibrary_t
    MtmlLibrary = mtmlLibraryInit(lib)

    # system 
    sys = c_mtmlSystem_t
    MtmlSystem = mtmlLibraryInitSystem(MtmlLibrary, sys)

    # version = mtmlSystemGetDriverVersion(MtmlSystem)
    # print("Driver Version:", version)

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

            mem = c_mtmlMemory_t
            mtmlDeviceInitMemory(MtmlDevice, mem)

            try:
                total_memory = mtmlMemoryGetTotal(mem)
                print("Total Memory:", total_memory, "MB")

                used_memory = mtmlMemoryGetUsed(mem)
                print("Used Memory:", used_memory, "MB")

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

        finally:
            mtmlLibraryFreeDevice(MtmlDevice)

    mtmlLibraryShutDown(MtmlLibrary)