
import struct
import numpy as np
import typing as tp
from pathlib import Path
from dataclasses import dataclass, field, fields

u8 = np.uint8
u16 = np.uint16
u32 = np.uint32
i32 = np.int32

HEADER_SIZE = 12

FORMATS = {
    i32: (4, '<i'),
    u32: (4, '<I'),
    u16: (2, '<H'),
    u8: (1, '<B')
}


class DataclassType(tp.Protocol):
    __dataclass_fields__: dict
    __dataclass_params__: dict
    __post_init__: tp.Optional[tp.Callable]


@dataclass
class Header:
    magic: str = field(metadata={'len': 4})
    dll_version: str = field(metadata={'len': 4})
    file_type: str = field(metadata={'len': 4})


@dataclass 
class OffsetTable:
    meas_conds: u32
    color_peak: u32
    color_light: u32
    light: list[u32] = field(metadata={'len': 3}) 
    height: list[u32] = field(metadata={'len': 3})
    clr_peak_thumb: u32
    clr_thumb: u32
    light_thumb: u32
    height_thumb: u32
    assembly_info: u32
    line_measure: u32
    line_thickness: u32
    string_data: u32
    reserved: u32


@dataclass
class MeasurementConditions:
    size: u32
    year: u32
    month: u32
    day: u32
    hour: u32
    minute: u32
    second: u32
    diff_from_UTC: i32
    img_attributes: u32
    user_interface_mode: u32
    color_composite_mode: u32
    img_layer_number: u32
    run_mode: u32
    peak_mode: u32
    sharpening_level: u32
    speed: u32
    distance: u32
    pitch: u32
    optical_zoom: u32
    number_of_lines: u32
    line0_pos: u32
    reserved_1: list[u32] = field(metadata={'len': 3}) 
    lens_magnification: u32
    PMT_gain_mode: u32
    PMT_gain: u32
    PMT_offset: u32
    ND_filter: u32
    reserved_2: u32
    persist_count: u32
    shutter_speed_mode: u32
    shutter_speed: u32
    white_balance_mode: u32
    white_balance_red: u32
    white_balance_blue: u32
    camera_gain: u32
    plane_compensation: u32
    xy_length_unit: u32
    z_length_unit: u32
    xy_decimal_place: u32
    z_decimal_place: u32
    x_length_per_pixel: u32
    y_length_per_pixel: u32
    z_length_per_digit: u32
    reserved_3: list[u32] = field(metadata={'len': 5}) 
    light_filter_type: u32
    reserved_4: u32
    gamma_reverse: u32
    gamma: u32
    gamma_correction_offset: u32
    CCD_BW_offset: u32
    num_aperature: u32
    head_type: u32
    PMG_gain_2: u32
    omit_color_image: u32
    lens_ID: u32
    light_lut_mode: u32
    light_lut_in0: u32
    light_lut_out0: u32
    light_lut_in1: u32
    light_lut_out1: u32
    light_lut_in2: u32
    light_lut_out2: u32
    light_lut_in3: u32
    light_lut_out3: u32
    light_lut_in4: u32
    light_lut_out4: u32
    upper_position: u32
    lower_position: u32
    light_effective_bit_depth: u32
    height_effective_bit_depth: u32


@dataclass
class AssemblyInformation:
    size: u32
    file_type: u16
    stage_type: u16
    x_position: u32
    y_position: u32


@dataclass
class AssemblyConditions:
    auto_adjustment: u8
    source: u8
    thin_out: u16
    count_x: u16
    count_y: u16


@dataclass
class AssemblyFile:
    source_files: list[u16] = field(metadata={'len': 260})
    pos_x: u8
    pos_y: u8
    datums_pos: u8
    fix_distance: u8
    distance_x: u32
    distance_y: u32


@dataclass
class ImageData:
    width: u32
    height: u32
    bit_depth: u32
    compression: u32
    data_byte_size: u32


@dataclass
class TrueColorImage(ImageData):
    data: np.ndarray = field(metadata={'dtype': np.uint8})


@dataclass
class FalseColorImage(ImageData):
    palette_range_min: u32
    palette_range_max: u32
    palette: np.ndarray = field(metadata={'dtype': np.uint8})
    data: np.ndarray = field(metadata={'dtype': np.uint32})


@dataclass
class LineMeasurement:
    size: u32
    line_width: u32
    light: list[u8] = field(metadata={'len': 3})
    height: list[u8] = field(metadata={'len': 3})


# @dataclass
# class CharacterStrings:
#     title: np.ndarray = field(metadata={'dtype': u8})
#     lens_name: np.ndarray = field(metadata={'dtype': u8})


@dataclass
class VK4Data:
    file_path: Path
    header: Header
    offset_table: OffsetTable
    measure_conds: MeasurementConditions
    color_peak: TrueColorImage
    color_light: TrueColorImage
    light: list[FalseColorImage]
    height: list[FalseColorImage]
    line_measure: LineMeasurement | None
    
    # assembly_info: AssemblyInformation
    # assembly_conds: AssemblyConditions
    # assembly_nfiles: u32
    # nimages: u32
    # assembly_files: list[AssemblyFile]
    # char_strings: CharacterStrings



class Vk4BinaryFile:
    def __init__(self, vk4_path: Path):
        self._idx = 0
        self._bytes = vk4_path.read_bytes()  

    def read_bytes(self, n_bytes: int):
        self._idx += n_bytes
        return self._bytes[self._idx-n_bytes:self._idx]

    def read_data(self, format='<I', size=4):
        return struct.unpack(format, self.read_bytes(size))[0]
    
    def read_dataclass[T](
        self,
        offset: int | u32,
        dataclass_type: tp.Type[T]
    ):
        self._idx = offset
        values = dict[str, tp.Any]()

        for field_ in fields(dataclass_type):   # type: ignore
            type_ = field_.type
            origin = tp.get_origin(type_)

            if origin == list:
                elem_type = tp.get_args(type_)[0]
                size, fmt = FORMATS[elem_type]

                values[field_.name] = [
                    self.read_data(fmt, size) 
                    for _ in range(field_.metadata['len'])
                ]

            elif type_ == str:
                str_len = field_.metadata['len']
                str_bytes: bytes = self.read_data(f'{str_len}s', str_len)
                values[field_.name] = str_bytes.decode('ascii')

            else:
                size, fmt = FORMATS[type_]
                values[field_.name] = self.read_data(fmt, size)

        return dataclass_type(**values)
    
    def read_image_data[T](
        self, 
        offset: int | u32, 
        dataclass_type: tp.Type[T],
        image_type: tp.Literal['color_peak', 'color_light', 'light', 'height'],
    )-> T:

        self._idx = offset

        width = self.read_data()
        height = self.read_data()
        bit_depth = self.read_data()
        compression = self.read_data()
        byte_size = self.read_data()

        N = width * height
        if 'color' in image_type:
            Z = bit_depth // 8
            data = self.read_bytes(N * Z)
            array = np.frombuffer(data, dtype=np.uint8)

            return dataclass_type(
                width,          # type: ignore              
                height, 
                bit_depth,                       
                compression, 
                byte_size, 
                array.copy().reshape((height, width, Z))
            ) 
        
        palette_min = self.read_data()
        palette_max = self.read_data()
        palette = np.frombuffer(self.read_bytes(768), dtype=np.uint8)

        dtype = np.dtype('<u2') if image_type == 'light' else np.dtype('<u4')
        array = np.frombuffer(self.read_bytes(N * dtype.itemsize), dtype=dtype)

        return dataclass_type(
            width,          # type: ignore
            height, 
            bit_depth, 
            compression, 
            byte_size,   
            palette_min, 
            palette_max, 
            palette.copy(), 
            array.copy().reshape((height, width))
        )


def extract_vk4_data(vk4_file: str):
    vk4_path = Path(vk4_file)
    assert vk4_path.exists() and vk4_path.name.endswith('.vk4')

    vk4 = Vk4BinaryFile(vk4_path)
    header = vk4.read_dataclass(0, Header)
    offsets = vk4.read_dataclass(HEADER_SIZE, OffsetTable)
    conditions = vk4.read_dataclass(offsets.meas_conds, MeasurementConditions)

    color_peak = vk4.read_image_data(offsets.color_peak, TrueColorImage, 'color_peak')
    color_light = vk4.read_image_data(offsets.color_light, TrueColorImage, 'color_light')

    light_images = [
        vk4.read_image_data(offset, FalseColorImage, 'light')
        for offset in offsets.light if offset > 0
    ]

    height_images = [
        vk4.read_image_data(offset, FalseColorImage, 'height')
        for offset in offsets.height if offset > 0
    ]

    line_measure = None
    if offsets.line_measure != 0:
        line_measure = vk4.read_dataclass(offsets.line_measure, LineMeasurement)

    return VK4Data(
        vk4_path,
        header,
        offsets,
        conditions,
        color_peak,
        color_light,
        light_images,
        height_images,
        line_measure
    ) 