from __future__ import annotations

import asyncio
import logging
import mmap
import random
import string
import struct
import time
from collections import namedtuple
import datetime
from decimal import Decimal
from enum import IntEnum
from pathlib import Path
from sys import maxsize as sys_maxsize
from typing import Optional, IO, Union, TYPE_CHECKING, Tuple, Any, Dict, List, Generator

import numpy as np

from ...Log import CLIENT_LOGGER_NAME

if TYPE_CHECKING:
    from ...Models.config import GlobalConfig, APIPullMethod, ZMSPullMethod


logger = logging.getLogger(CLIENT_LOGGER_NAME)
LP = "media:"
g: Optional[GlobalConfig] = None


class PipeLine:
    _event_data_recursions: int = 0
    options: Union[APIPullMethod, ZMSPullMethod, None] = None
    event_tot_frames: int = 0
    attempted_fids: List[int] = list()
    start_datetime: Optional[datetime] = None
    event_end_datetime: Optional[datetime] = None
    event_tot_seconds: Optional[float] = None
    increment_by: Optional[Union[int, float]] = None
    current_frame: int = 1

    async def get_event_data(self, msg: Optional[str] = None):
        """Calls global DB get event data"""
        if msg:
            logger.debug(f"{LP}read>event_data: {msg}")

        if not self.event_ended or not g.Frame:
            try:
                await g.db.get_all_event_data(g.eid)

            except Exception as e:
                logger.error(f"{LP} error grabbing event data from DB -> {e}")
                # recurse
                if self._event_data_recursions < 3:
                    self._event_data_recursions += 1
                    await self.get_event_data(msg="retrying to grab event data...")
                else:
                    logger.error(
                        f"{LP} max recursions reached trying to grab event data, aborting!"
                    )
                    raise RuntimeError(
                        f"{LP} max recursions reached trying to grab event data, aborting!"
                    )
            else:
                self.event_tot_frames = len(g.Frame)
                self.event_end_datetime = g.db.end_datetime_from_eid(g.eid)
                self.start_datetime = g.db.start_datetime_from_eid(g.eid)
                self.event_tot_seconds = g.Event.get(
                    "Length",
                    (self.event_end_datetime - self.start_datetime).total_seconds(),
                )
                self.increment_by = int(self.event_tot_frames / self.event_tot_seconds)
                logger.debug(
                    f"{LP} grabbed event data from ZM API for event '{g.eid}' -- event total Frames: "
                    f"{self.event_tot_frames} -- EndDateTime: {self.event_end_datetime} -- StartDateTime: "
                    f"{self.start_datetime} - EventTotSec: {self.event_tot_seconds} - IncrementBy: {self.increment_by} "
                    f"- has event ended: {self.event_ended}"
                )
        else:
            logger.debug(f"{LP} event has ended, no need to grab event data")

    def __init__(self):
        global g
        from ...main import get_global_config

        g = get_global_config()

    @property
    def event_ended(self):
        if self.event_end_datetime:
            return True
        return False

    @property
    def frames_attempted(self):
        return len(self.attempted_fids) or 0

    def _process_frame(
        self,
        image: bytes = None,
        end: bool = False,
    ) -> Tuple[Optional[Union[bytes, bool]], Optional[str]]:
        """Process the frame, increment counters, and return the image if there is one"""
        lp = f"{LP}process_frame:"
        self.attempted_fids.append(self.current_frame)
        if end:
            logger.error(f"{lp} end has been called, no more images to process!")
            return False, None

        self.current_frame = self.current_frame + self.increment_by
        logger.debug(
            f"{lp} incrementing next frame ID to read by {self.increment_by} = {self.current_frame}"
        )
        if image:
            img_name = f"mid_{g.mid}_random_{random.randint(0,1000)}.jpg"
            # (bytes, image_file_name)
            return (
                image,
                img_name,
            )
        return None, None

    async def image_generator(
        self,
    ) -> Generator[Tuple[Optional[Union[bytes, bool]], Optional[str]]]:
        """Generator to return images from the source"""
        logger.debug(f"{LP}image_generator: STARTING -- {g.past_event=}")
        while True:
            yield await self.get_image()

    async def get_image(self) -> Tuple[Optional[Union[bytes, bool]], Optional[str]]:
        raise NotImplementedError


class APIImagePipeLine(PipeLine):
    """An image grabber that uses ZoneMinders API as its source"""

    from ...Models.config import APIPullMethod

    def __init__(
        self,
        options: APIPullMethod,
    ):
        lp = f"{LP}API:init::"
        assert options, f"{lp} no stream options provided!"
        super().__init__()
        #  INIT START 
        self.options = options
        logger.debug(f"{lp} options: {self.options}")
        self.has_event_ended: str = ""
        self.max_attempts = options.attempts
        if g.past_event:
            logger.debug(f"{lp} this is a past event, max image grab attempts set to 1")
            self.max_attempts = 1
        self.max_attempts_delay = options.delay
        self.sbf: Optional[int] = self.options.sbf
        self.fps: Optional[int] = self.options.fps
        self.skip_frames_calc: int = 0
        self.event_end_datetime: str = ""

        # Alarm frame is always the first frame (the frame that kicked the event off)
        # pre count buffer length+1 for alarm frame
        self.current_frame = self.buffer_pre_count + 1
        # The pre- / post-buffers will give the absolute minimum number of frames to grab, assuming no event issues
        self.total_min_frames = int(
            (self.buffer_post_count + self.buffer_pre_count) / self.capture_fps
        )
        # We don't know how long an event will be so set an upper limit of at least
        # pre- + post-buffers calculated as seconds because we are pulling <X> FPS
        self.total_max_frames = self.options.max_frames

    async def get_image(self) -> Tuple[Optional[Union[bytes, bool]], Optional[str]]:
        if self.frames_attempted >= self.total_max_frames:
            logger.error(
                f"max_frames ({self.total_max_frames}) has been reached, stopping!"
            )
            return False, None

        import aiohttp

        response: Optional[aiohttp.ClientResponse] = None
        lp = f"{LP}read:"
        if self.frames_attempted > 0:
            logger.debug(
                f"{lp} [{self.frames_processed}/{self.total_max_frames} frames processed: {self._processed_fids}] "
                f"- [{self.frames_skipped}/{self.total_max_frames} frames skipped: {self._skipped_fids}] - "
                f"[{self.frames_attempted}/{self.total_max_frames} frames attempted: {self.attempted_fids}]"
            )
        else:
            logger.debug(f"{lp} processing first frame!")
            _msg = f"{lp} checking snapshot ids enabled, will check every {self.options.snapshot_frame_skip} frames"
            if g.past_event:
                _msg = (
                    f"{lp} this is a past event (not live), skipping snapshot "
                    f"id checks (snapshot frame ID will not change)"
                )
            logger.debug(_msg) if self.options.check_snapshots else None

        curr_snapshot = None
        if self.options.check_snapshots:
            # Only run every <x> frames, if it's a live event
            if (
                (not g.past_event)
                and (self.frames_processed > 0)
                and (self.frames_processed % self.options.snapshot_frame_skip == 0)
            ):
                await self.get_event_data(
                    msg=f"grabbing data for snapshot comparisons..."
                )
                if curr_snapshot := int(g.Event.get("MaxScoreFrameId", 0)):
                    if self.last_snapshot_id:
                        if curr_snapshot > self.last_snapshot_id:
                            logger.debug(
                                f"{lp} current snapshot frame id is not the same as the last snapshot id "
                                f"CURR:{curr_snapshot} - PREV:{self.last_snapshot_id}, grabbing new snapshot image"
                            )
                            self.current_frame = curr_snapshot
                        else:
                            logger.debug(
                                f"{lp} current snapshot frame id is the same as the last snapshot id "
                                f"CURR:{curr_snapshot} - PREV:{self.last_snapshot_id}, skipping frame"
                            )
                    self.last_snapshot_id = curr_snapshot
                else:
                    logger.warning(
                        f"{lp} Event: {g.eid} - No Snapshot Frame ID found in ZM API? -> {g.Event = }",
                    )

        #  Check if we have already processed this frame ID 
        if self.current_frame in self._processed_fids:
            logger.debug(
                f"{lp} skipping Frame ID: '{self.current_frame}' as it has already been"
                f" processed for event {g.eid}"
            )
            return self._process_frame(skip=True)
        #  SET URL TO GRAB IMAGE FROM 
        logger.debug(f"Calculated Frame ID as: {self.current_frame}")
        portal_url = str(g.api.portal_base_url)
        if portal_url.endswith("/"):
            portal_url = portal_url[:-1]
        fid_url = (
            f"{portal_url}/index.php?view=image&eid={g.eid}&fid={self.current_frame}"
        )
        timeout = g.config.zoneminder.pull_method.api.timeout or 15

        for image_grab_attempt in range(self.max_attempts):
            image_grab_attempt += 1
            logger.debug(
                f"{lp} attempt #{image_grab_attempt}/{self.max_attempts} to grab image ID: {self.current_frame}"
            )
            _perf = time.time()
            api_response = await g.api.make_async_request(fid_url, timeout=timeout)
            logger.debug(
                f"perf:{lp} API request took {time.time() - _perf:.5f)} seconds"
            )
            if isinstance(api_response, bytes) and api_response.startswith(
                b"\xff\xd8\xff"
            ):
                logger.debug(f"ZM API returned a JPEG formatted image!")
                return self._process_frame(image=api_response)
            else:
                resp_msg = ""
                if api_response:
                    resp_msg = f" response code={api_response.status} - response={api_response}"
                else:
                    resp_msg = f" no response received!"

                logger.warning(f"{lp} image was not retrieved!{resp_msg}")

                await self.get_event_data(msg="checking if event has ended...")

                if self.event_ended:  # Assuming event has ended
                    logger.debug(f"{lp} event has ended, checking OOB status...")
                    # is current frame OOB
                    if self.current_frame > self.event_tot_frames:
                        # We are OOB, so we are done
                        logger.debug(
                            f"{lp} we are OOB in a FINISHED event (current requested fid: {self.current_frame} > "
                            f"total frames in event: {self.event_tot_frames})"
                        )
                        return self._process_frame(end=True)
                else:
                    logger.debug(
                        f"{lp} event has not ended yet! Total Frames: {self.event_tot_frames}"
                    )
                if not g.past_event and (image_grab_attempt < self.max_attempts):
                    logger.debug(f"{lp} sleeping for {self.options.delay} second(s)")
                    time.sleep(self.options.delay)

        return self._process_frame(skip=True)

    @property
    def frames_processed(self):
        return len(self.attempted_fids) or 0


class ZMSImagePipeLine(PipeLine):
    max_frames: Optional[int] = 0
    """
    This image pipeline is designed to work with ZM CGI script nph-zms.
    nph = No Parsed Headers
    **nph-zms is symlinked to zms**

    http://localhost/zm/cgi-bin/nph-zms?mode=single&monitor=1&user=USERNAME&pass=PASSWORD"
    works with token='<ACCESS TOKEN>' as well
    mode=jpeg or single
    monitor=<mid> will ask for monitor mode
    event=<eid> will ask for event mode
    frame=<fid> will ask for a specific frame from an event (implies event mode)
    """

    def __init__(
        self,
        options: ZMSPullMethod,
    ):
        lp = f"{LP}ZMS:init::"
        assert options, f"{lp} no stream options provided!"
        super().__init__()
        #  INIT START 
        self.options = options
        self.url: Optional[str] = str(options.url) if options.url else None
        logger.debug(f"{lp} options: {self.options}")

        self.max_attempts = 1
        self.max_attempts = options.attempts
        self.max_attempts_delay = options.delay
        self.max_frames = options.max_frames
        # Process URL, if it is empty grab API portal and append default path
        if not self.url:
            logger.debug(
                f"{lp} no URL provided, constructing from API portal and ZMS_CGI_PATH from zm.conf"
            )
            cgi_sys_path = Path(g.db.cgi_path)
            # ZM_PATH_CGI=/usr/lib/zoneminder/cgi-bin
            portal_url = str(g.api.portal_base_url)
            if portal_url.endswith("/"):
                portal_url = portal_url[:-1]
            self.url = f"{portal_url}/{cgi_sys_path.name}/nph-zms"

        if g.past_event:
            # if this is a past (non live) event, grab event data
            # use a task so we don't block the main thread
            import asyncio

            loop = asyncio.get_running_loop()
            loop.create_task(self.get_event_data())

    async def get_image(self) -> Tuple[Optional[Union[bytes, bool]], Optional[str]]:
        if self.frames_attempted >= self.max_frames:
            logger.error(f"max_frames ({self.max_frames}) has been reached, stopping!")
            return False, None

        lp = f"{LP}ZMS:read:"
        timeout = g.config.zoneminder.pull_method.zms.timeout or 15
        letters = string.ascii_lowercase

        if not g.past_event:
            # Live event - Pull images from live stream
            url = f"{self.url}?mode=single&monitor={g.mid}&connkey={random.randint(100000, 999999)}"
            _sep = "-_/.,`~<>="
            rand_str = "".join(random.choice(letters) for i in range(8))
            rand_int = random.randint(1, 999999)
            sep = random.choice(_sep)
            self.attempted_fids.append(f"{rand_str}{sep}{rand_int}")

            start_img_req = time.time()
            api_response = await g.api.make_async_request(
                url=url, type_action="post", timeout=timeout
            )
            end_img_req = time.time() - start_img_req
            logger.debug(f"perf:{lp} ZMS request took: {end_img_req:.5f}")
            return_img = None
            img_reason = ""
            if not api_response:
                img_reason = f" no response received!"

            elif isinstance(api_response, bytes):
                if api_response.startswith(b"\xff\xd8\xff"):
                    logger.debug(f"{lp} Response is a JPEG formatted image!")
                    return_img = api_response
                # else:
                #     logger.debug(
                #         f"{lp} bytes data returned -> {api_response}"
                #     )

            else:
                img_reason = f" response is not bytes -> Type: {type(api_response)} -- {api_response = }"

            if return_img:
                return return_img, f"mid_{g.mid}_rand_{random.randint(0,1000)}.jpg"

            logger.warning(f"{lp} image was not retrieved!{img_reason}")

            return None, None

        else:
            # Past event, iterate event using frame id's (target 1 fps)
            if self.frames_attempted > 0:
                pass
            else:
                logger.debug(f"{lp} processing first frame!")

            url = f"{self.url}?mode=jpeg&event={g.eid}&frame={self.current_frame}&connkey={random.randint(100000, 999999)}"
            past_perf = time.time()

            for image_grab_attempt in range(self.max_attempts):
                image_grab_attempt += 1
                past_or_live = "past" if g.past_event else "live"
                pol2 = "event data" if g.past_event else "frame buffer"
                logger.debug(
                    f"{lp} attempt #{image_grab_attempt}/{self.max_attempts} to grab {past_or_live} image from mid: "
                    f"{g.mid} {pol2}"
                )
                api_response = await g.api.make_async_request(
                    url=url, type_action="post", timeout=timeout
                )
                # logger.debug(f"{lp} URL: {url}")
                end_perf = time.time()
                logger.debug(f"perf:{lp} ZMS request took: {end_perf - past_perf:.5f}")
                resp_msg = ""
                # Cover unset and None
                if not api_response:
                    # if isinstance(api_response, aiohttp.ClientResponse):
                    #     resp_msg = f"<<response code={api_response.status}>> - response={api_response}"
                    resp_msg = f"no response received!"

                elif isinstance(api_response, bytes):
                    if api_response.startswith(b"\xff\xd8\xff"):
                        logger.debug(f"{lp} Response is a JPEG formatted image!")
                        return self._process_frame(image=api_response)
                else:
                    resp_msg = f"{lp} response is not bytes -> {type(api_response) = } -- {api_response = }"

                if resp_msg:
                    str_delay = ''
                    if self.max_attempts_delay:
                        str_delay = f" - sleeping for {self.max_attempts_delay} seconds"
                    logger.debug(
                        f"{lp} No image returned ({resp_msg}){str_delay}"
                    )
                    if self.max_attempts_delay:
                        await asyncio.sleep(self.max_attempts_delay)

                return self._process_frame()


# class SHMImagePipeLine(PipeLine):
#     class ZMAlarmStateChanges(IntEnum):
#         STATE_IDLE = 0
#         STATE_PREALARM = 1
#         STATE_ALARM = 2
#         STATE_ALERT = 3
#         STATE_TAPE = 4
#         ACTION_GET = 5
#         ACTION_SET = 6
#         ACTION_RELOAD = 7
#         ACTION_SUSPEND = 8
#         ACTION_RESUME = 9
#         TRIGGER_CANCEL = 10
#         TRIGGER_ON = 11
#         TRIGGER_OFF = 12
#
#     def __init__(self):
#         path = "/dev/shm"
#         # ascertain where the SHM filesystem is mounted
#         if not Path(path).exists():
#             path = "/run/shm"
#             if not Path(path).exists():
#                 raise FileNotFoundError(
#                     f"Cannot find SHM filesystem at /dev/shm or /run/shm"
#                 )
#
#         self.IS_64BITS = sys_maxsize > 2**32
#         # This will compensate for 32/64 bit
#         self.struct_time_stamp = r"l"
#         self.TIMEVAL_SIZE: int = struct.calcsize(self.struct_time_stamp)
#         self.alarm_state_stages = self.ZMAlarmStateChanges
#         self.file_handle: Optional[IO] = None
#         self.mem_handle: Optional[mmap.mmap] = None
#
#         self.file_name = f"{path}/zm.mmap.{g.mid}"
#
#     def reload(self):
#         """Reloads monitor information. Call after you get
#         an invalid memory report
#
#         Raises:
#             ValueError: if no monitor is provided
#         """
#         # close file handler
#         self.close()
#         # open file handler in read binary mode
#         self.file_handle = open(self.file_name, "r+b")
#         # geta rough size of the memory consumed by object (doesn't follow links or weak ref)
#         from os.path import getsize
#
#         sz = getsize(self.file_name)
#         if not sz:
#             raise ValueError(f"Invalid size: {sz} of {self.file_name}")
#
#         self.mem_handle = mmap.mmap(
#             self.file_handle.fileno(), 0, access=mmap.ACCESS_READ
#         )
#         self.sd = None
#         self.td = None
#         self.get_image()
#
#     def is_valid(self):
#         """True if the memory handle is valid
#
#         Returns:
#             bool: True if memory handle is valid
#         """
#         try:
#             d = self.get_image()
#             return not d["shared_data"]["size"] == 0
#         except Exception as e:
#             logger.debug(f"ERROR!!!! =-> Memory: {e}")
#             return False
#
#     def get_image(self):
#         self.mem_handle.seek(0)  # goto beginning of file
#         # import proper class that contains mmap data
#
#         sd_model = x.shared_data
#         td_model = x.trigger_data
#         vs_model = x.video_store_data
#
#         SharedData = sd_model.named_tuple
#
#         # old_shared_data = r"IIIIQIiiiiii????IIQQQ256s256s"
#         # I = uint32 - i = int32 == 4 bytes ; int
#         # Q = uint64 - q = int64 == 8 bytes ; long long int
#         # L = uint64 - l = int64 == 8 bytes ; long int
#         # d = double == 8 bytes ; float
#         # s = char[] == n bytes ; string
#         # B = uint8 - b = int8 == 1 byte; char
#
#         # shared data bytes is now aligned at 776 as of commit 590697b (1.37.19) -> SEE
#         # https://github.com/ZoneMinder/zoneminder/commit/590697bd807ab9a74d605122ef0be4a094db9605
#         # Before it was 776 for 64bit and 772 for 32 bit
#
#         TriggerData = td_model.named_tuple
#
#         VideoStoreData = vs_model.named_tuple
#
#         s = SharedData._make(
#             struct.unpack(sd_model.struct_str, self.mem_handle.read(sd_model.bytes))
#         )
#         t = TriggerData._make(
#             struct.unpack(td_model.struct_str, self.mem_handle.read(td_model.bytes))
#         )
#         v = VideoStoreData._make(
#             struct.unpack(vs_model.struct_str, self.mem_handle.read(vs_model.bytes))
#         )
#
#         written_images = s.last_write_index + 1
#         timestamp_offset = s.size + t.size + v.size
#         images_offset = timestamp_offset + g.mon_image_buffer_count * self.TIMEVAL_SIZE
#         # align to nearest 64 bytes
#         images_offset = images_offset + 64 - (images_offset % 64)
#         # images_offset = images_offset + s.imagesize * s.last_write_index
#         # Read timestamp data
#         ts_str = " "
#         if s.last_write_index <= g.mon_image_buffer_count >= 0:
#             for loop in range(written_images):
#                 ts_str = f"image{loop + 1}_ts"
#         self.mem_handle.seek(timestamp_offset)
#         TimeStampData = namedtuple(
#             "TimeStampData",
#             ts_str,
#         )
#         ts = TimeStampData._make(
#             struct.unpack(
#                 self.struct_time_stamp,
#                 self.mem_handle.read(g.mon_image_buffer_count * self.TIMEVAL_SIZE),
#             )
#         )
#         logger.debug(f"{written_images = } - {images_offset = }")
#         logger.debug(f"\nSharedData = {s}\n")
#         self.mem_handle.seek(images_offset)
#         # only need 1 image, not the recent buffers worth
#         image_buffer = self.mem_handle.read(int(s.imagesize))
#         # image_buffer = self.mem_handle.read(written_images * s.imagesize)
#         import cv2
#         import numpy as np
#
#         img: Optional[np.ndarray] = None
#         logger.debug(f"Converting images into ndarray")
#         # grab total image buffer
#         # convert bytes to numpy array to cv2 images
#         # for i in range(written_images):
#         # img = np.frombuffer(image_buffer[i * s.imagesize : (i + 1) * s.imagesize], dtype=np.uint8)
#         img = np.frombuffer(image_buffer, dtype=np.uint8)
#         if img.size == s.imagesize:
#             logger.debug(f"Image is of the correct size, reshaping and converting")
#             img = img.reshape((g.mon_height, g.mon_width, g.mon_colorspace))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         else:
#             logger.debug(f"Invalid image size: {img.size}")
#         # show images
#         self.sd = s._asdict()
#         self.td = t._asdict()
#         self.vsd = v._asdict()
#         self.tsd = ts._asdict()
#
#         self.sd["video_fifo_path"] = (
#             self.sd["video_fifo_path"].decode("utf-8").strip("\x00")
#         )
#         self.sd["audio_fifo_path"] = (
#             self.sd["audio_fifo_path"].decode("utf-8").strip("\x00")
#         )
#         self.sd["janus_pin"] = self.sd["janus_pin"].decode("utf-8").strip("\x00")
#         self.vsd["event_file"] = self.vsd["event_file"].decode("utf-8").strip("\x00")
#         self.td["trigger_text"] = self.td["trigger_text"].decode("utf-8").strip("\x00")
#         self.td["trigger_showtext"] = (
#             self.td["trigger_showtext"].decode("utf-8").strip("\x00")
#         )
#         self.td["trigger_cause"] = (
#             self.td["trigger_cause"].decode("utf-8").strip("\x00")
#         )
#         self.sd["alarm_cause"] = self.sd["alarm_cause"].decode("utf-8").strip("\x00")
#         self.sd["control_state"] = (
#             self.sd["control_state"].decode("utf-8").strip("\x00")
#         )
#
#         return {
#             "shared_data": self.sd,
#             "trigger_data": self.td,
#             "video_store_data": self.vsd,
#             "time_stamp_data": self.tsd,
#             "image": img,
#         }
#
#     def close(self):
#         """Closes the handle"""
#         try:
#             if self.mem_handle:
#                 self.mem_handle.close()
#             if self.file_handle:
#                 self.file_handle.close()
#         except Exception:
#             pass


class ZMUImagePipeLine(PipeLine):
    offset = None


class FileImagePipeLine(PipeLine):
    config: Optional[Dict[str, Any]] = None
    image: Optional[np.ndarray] = None
    input_file: Optional[Path] = None
    video: Optional[Any] = None


def zm_version(
    ver: str, minx: Optional[int] = None, patchx: Optional[int] = None
) -> int:
    maj, min, patch = "", "", ""
    x = ver.split(".")
    x_len = len(x)
    if x_len <= 2:
        maj, min = x
        patch = "0"
    elif x_len == 3:
        maj, min, patch = x
    else:
        logger.debug("come and fix me!?!?!")
    maj = int(maj)
    min = int(min)
    patch = int(patch)
    if minx:
        if minx > min:
            return 1
        elif minx == min:
            if patchx:
                if patchx > patch:
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return 0
