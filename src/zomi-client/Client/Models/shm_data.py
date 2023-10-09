"""This stores the MMAP data needed to determine image offsets in SHM"""
from collections import namedtuple
import logging

from pydantic import BaseModel

from ..Log import CLIENT_LOGGER_NAME

logger = logging.getLogger(CLIENT_LOGGER_NAME)


class SharedBase(BaseModel):
    """Base class for shm data structs"""

    bytes: int = None
    named_tuple: namedtuple = None
    struct_str: str = None


class Dot3725:
    """Data Structure for 1.37.25"""

    def __init__(self):
        self.shared_data: SharedBase = SharedBase(
            bytes=840,
            named_tuple=namedtuple(
                "SharedData",
                "size last_write_index last_read_index state capture_fps analysis_fps last_event_id action brightness "
                "hue colour contrast alarm_x alarm_y valid capturing analysing recording signal format imagesize "
                "last_frame_score audio_frequency audio_channels startup_time zmc_heartbeat_time last_write_time "
                "last_read_time last_viewed_time control_state alarm_cause video_fifo_path audio_fifo_path, "
                "janus_pin",
            ),
            struct_str=r"I 2i I 2d Q I 6i 6B 4I 5L 256s 256s 64s 64s 64s",
        )
        self.trigger_data: SharedBase = SharedBase(
            bytes=560,
            named_tuple=namedtuple(
                "TriggerData",
                "size trigger_state trigger_score padding trigger_cause trigger_text trigger_showtext",
            ),
            struct_str=r"4I 32s 256s 256s",
        )
        self.video_store_data: SharedBase = SharedBase(
            bytes=4120,
            named_tuple=namedtuple(
                "VideoStoreData",
                "size current_event event_file recording",
            ),
            struct_str=r"I Q 4096s L",
        )
