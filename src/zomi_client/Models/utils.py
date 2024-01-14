from __future__ import annotations
import logging
from hashlib import new
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, TYPE_CHECKING, AnyStr
import warnings

import cv2
import numpy as np
from pydantic import SecretStr
import requests
try:
    from shapely.geometry import Polygon
except ImportError as e:
    warnings.warn(f"ImportError: {e}", ImportWarning)
    Polygon = None

from ..Log import CLIENT_LOGGER_NAME
from .validators import str2path

if TYPE_CHECKING:
    from ..Libs.API import ZMAPI
    from ..Models.config import GlobalConfig

g: Optional[GlobalConfig] = None
logger = logging.getLogger(CLIENT_LOGGER_NAME)

SLATE_COLORS: List[Tuple[int, int, int]] = [
    (39, 174, 96),
    (142, 68, 173),
    (0, 129, 254),
    (254, 60, 113),
    (243, 134, 48),
    (91, 177, 47),
]


def draw_filtered_bboxes(
    image: np.ndarray,
    filtered_result: List[str, float, List[int]],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draws the bounding boxes on the image.

    Args:
        image (np.ndarray): The image to draw on.
        filtered_result (List[str, float, List[int]]): The filtered bounding boxes.
        color (Tuple[int, int, int]): The color to use [Default: Red].
        thickness (int, optional): The thickness of the bounding boxes. Defaults to 2.

    Returns:
        np.ndarray: The image with the bounding boxes drawn.
    """
    # image = image.copy()
    lp = f"image::draw filtered bbox::"
    w, h = image.shape[:2]

    for label, conf, bbox in filtered_result:
        logger.debug(f"{lp} drawing '{label}' bounding box @ {bbox}")
        # draw bounding box around object
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

        # write label
        font_thickness = 1
        font_scale = 0.6
        # FIXME: add something better than this
        if w >= 720:
            # 720p+
            font_scale = 1.0
            font_thickness = 2
        elif w >= 1080:
            # 1080p+
            font_scale = 1.7
            font_thickness = 2
        elif w >= 1880:
            # 3-4k ish? +
            font_scale = 3.2
            font_thickness = 4

        font_type = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(label, font_type, font_scale, font_thickness)[0]
        text_width_padded = text_size[0] + 4
        text_height_padded = text_size[1] + 4
        r_top_left = (bbox[0], bbox[1] - text_height_padded)
        r_bottom_right = (bbox[0] + text_width_padded, bbox[1])
        # Background
        cv2.rectangle(image, r_top_left, r_bottom_right, color, -1)
        # Make sure the text is inside the image
        if r_bottom_right[0] > w:
            r_bottom_right = (w, r_bottom_right[1])
            r_top_left = (r_bottom_right[0] - text_width_padded, r_top_left[1])
        if r_bottom_right[1] > h:
            r_bottom_right = (r_bottom_right[0], h)
            r_top_left = (r_top_left[0], r_bottom_right[1] - text_height_padded)

        cv2.putText(
            image,
            label,
            (bbox[0] + 2, bbox[1] - 2),
            font_type,
            font_scale,
            [255, 255, 255],
            font_thickness,
            cv2.LINE_AA,
        )
    return image


def font_scale_from_image_size(image: np.ndarray) -> float:
    """Returns a font scale and thickness based on the image size.

    Args:
        image (np.ndarray): The image to get the font scale from.

    Returns:
        Tuple[float, int]: The font scale, thickness.
    """
    w, h = image.shape[:2]
    font_thickness = 1
    font_scale = 0.6
    # FIXME: add something better than this
    if w >= 720:
        # 720p+
        font_scale = 1.0
        font_thickness = 2
    elif w >= 1080:
        # 1080p+
        font_scale = 1.7
        font_thickness = 2
    elif w >= 1880:
        # 3-4k ish? +
        font_scale = 3.2
        font_thickness = 4
    return font_scale, font_thickness


def draw_zones(
    image: np.ndarray,
    zone_polygons: List[Dict[AnyStr, Polygon]],
    poly_color: Tuple[int, int, int] = (0, 0, 255),
    poly_line_thickness: int = 1,
    show_name: bool = False
):
    lp: str = "image::zones::"
    # image = image.copy()
    if not poly_line_thickness:
        poly_line_thickness = 1

    for zone_polygon in zone_polygons:
        zone_name = zone_polygon["name"]
        zone_pts = zone_polygon["points"]
        try:
            raw_pts = list(zip(*zone_pts.exterior.coords.xy))[:-1]
            # shape of point array [len(raw_points), 2]
            _pts = np.asarray(raw_pts).astype(np.int32)
            # reshape the point array to make it 3D
            _pts = _pts.reshape(-1, 1, 2)  # now shape [len(raw_points), 1, 2]
            cv2.polylines(
                image,
                [_pts],
                True,
                poly_color,
                poly_line_thickness,
            )
        except Exception as exc:
            logger.error(f"{lp} could not draw zone '{zone_name}' polygon -> {exc}", exc_info=True)
            return
        else:
            logger.debug(f"{lp} Successfully drew zone '{zone_name}' polygon")
            if show_name:
                # label the zone
                font_scale, font_thickness = font_scale_from_image_size(image)
                font_type = cv2.FONT_HERSHEY_DUPLEX
                text_size = cv2.getTextSize(
                    zone_name, font_type, font_scale, font_thickness
                )[0]
                text_width_padded = text_size[0] + 4
                text_height_padded = text_size[1] + 4
                r_top_left = (
                    int(zone_pts.centroid.x),
                    int(zone_pts.centroid.y - text_height_padded),
                )
                r_bottom_right = (
                    int(zone_pts.centroid.x + text_width_padded),
                    int(zone_pts.centroid.y),
                )
                try:
                    # # Make sure the text is inside the image
                    # if r_bottom_right[0] > image.shape[1]:
                    #     r_bottom_right = (image.shape[1], r_bottom_right[1])
                    #     r_top_left = (
                    #         r_bottom_right[0] - text_width_padded,
                    #         r_top_left[1],
                    #     )
                    # if r_bottom_right[1] > image.shape[0]:
                    #     r_bottom_right = (r_bottom_right[0], image.shape[0])
                    #     r_top_left = (
                    #         r_top_left[0],
                    #         r_bottom_right[1] - text_height_padded,
                    #     )
                    # Background
                    cv2.rectangle(image, r_top_left, r_bottom_right, (0, 0, 0), -1)
                    # adjust text to be on background
                    text_pos = (r_top_left[0] + 2, r_bottom_right[1] - 2)
                    cv2.putText(
                        image,
                        text=zone_name,
                        org=text_pos,
                        fontFace=font_type,
                        fontScale=font_scale,
                        color=[255, 255, 255],
                        thickness=font_thickness,
                        lineType=cv2.LINE_AA,
                        bottomLeftOrigin=False,
                    )
                except Exception as exc:
                    logger.error(f"{lp} could not label zone name: '{zone_name}' -> {exc}", exc_info=True)
                    return
                else:
                    logger.debug(f"{lp} Successfully labeled zone name: '{zone_name}'")
    return image


def draw_bounding_boxes(
    image: np.ndarray,
    labels: List[str],
    confidences: List[float],
    boxes: List[List[int]],
    model: str,
    processor: str,
    write_conf: bool = False,
    write_model: bool = False,
    write_processor: bool = False,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image"""
    # FIXME: need to add scaling dependant on image dimensions
    bgr_slate_colors = SLATE_COLORS[::-1]
    w, h = image.shape[:2]
    lp = f"image:draw bbox::"
    arr_len = len(bgr_slate_colors)
    i = 0
    for label, conf, bbox in zip(labels, confidences, boxes):
        logger.debug(f"{lp} drawing '{label}' bounding box @ {bbox}")
        bbox_color = bgr_slate_colors[i % arr_len]
        i += 1
        if write_conf and conf:
            label = f"{label} {round(conf * 100)}%"
        if write_model and model:
            if write_processor and processor:
                label = f"{label} [{model}::{processor}]"

            else:
                label = f"{label} [{model}]"
            logger.debug(f"{lp} writing model '{model}'")
        # draw bounding box around object
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 2)

        # write label
        font_scale, font_thickness = font_scale_from_image_size(image)
        font_type = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(label, font_type, font_scale, font_thickness)[0]
        text_width_padded = text_size[0] + 4
        text_height_padded = text_size[1] + 4
        r_top_left = (bbox[0], bbox[1] - text_height_padded)
        r_bottom_right = (bbox[0] + text_width_padded, bbox[1])
        # Background
        cv2.rectangle(image, r_top_left, r_bottom_right, bbox_color, -1)
        # Make sure the text is inside the image
        if r_bottom_right[0] > w:
            r_bottom_right = (w, r_bottom_right[1])
            r_top_left = (r_bottom_right[0] - text_width_padded, r_top_left[1])
        if r_bottom_right[1] > h:
            r_bottom_right = (r_bottom_right[0], h)
            r_top_left = (r_top_left[0], r_bottom_right[1] - text_height_padded)

        cv2.putText(
            image,
            label,
            (bbox[0] + 2, bbox[1] - 2),
            font_type,
            font_scale,
            [255, 255, 255],
            font_thickness,
            cv2.LINE_AA,
        )

    return image


class CFGHash:
    previous_hash: str
    config_file: Path
    hash: str

    def __init__(self, config_file: Union[str, Path, None] = None):
        self.previous_hash = ""
        lp = "hash::init::"
        self.hash = ""
        if config_file:
            self.config_file = str2path(config_file)

    def compute(
        self,
        input_file: Optional[Union[str, Path]] = None,
        read_chunk_size: int = 65536,
        algorithm: str = "sha256",
    ):
        """Hash a file using hashlib.
        Default algorithm is SHA-256

        :param input_file: File to hash
        :param int read_chunk_size: Maximum number of bytes to be read from the file
         at once. Default is 65536 bytes or 64KB
        :param str algorithm: The hash algorithm name to use. For example, 'md5',
         'sha256', 'sha512' and so on. Default is 'sha256'. Refer to
         hashlib.algorithms_available for available algorithms
        """

        lp: str = f"hash::compute::{algorithm}::"
        checksum = new(algorithm)  # Raises appropriate exceptions.
        self.previous_hash = str(self.hash)
        self.hash = ""

        if input_file:
            logger.debug(f"{lp} input_file={input_file}")
            self.config_file = str2path(input_file)

        try:
            with self.config_file.open("rb") as f:
                for chunk in iter(lambda: f.read(read_chunk_size), b""):
                    checksum.update(chunk)
        except Exception as exc:
            logger.warning(
                f"{lp} ERROR while computing {algorithm} hash of "
                f"'{self.config_file.as_posix()}' -> {exc}"
            )
            raise
        else:
            self.hash = checksum.hexdigest()
            logger.debug(
                f"{lp} the hex-digest for file '{self.config_file.expanduser().resolve().as_posix()}' -> {self.hash}"
            )
        return self.hash

    def compare(self, compare_hash: str) -> bool:
        if self.hash == compare_hash:
            return True
        return False

    def __repr__(self):
        return f"{self.hash}"

    def __str__(self):
        return self.__repr__()



def get_push_auth(api: ZMAPI, user: SecretStr, pw: SecretStr, has_https: bool = False):
    from urllib.parse import urlencode, quote_plus

    lp = "get_api_auth::"
    push_auth = ""
    if user:
        logger.debug(f"{lp} user supplied...")
        if pw:
            logger.debug(f"{lp} password supplied...")
            if has_https:
                logger.debug(f"{lp} HTTPS detected, using user/pass in url")

                payload = {
                    "user": user.get_secret_value(),
                    "pass": pw.get_secret_value(),
                }
                push_auth = urlencode(payload, quote_via=quote_plus)
            elif not has_https:
                logger.warning(
                    f"{lp} HTTP detected, using token (tokens expire, therefore "
                    f"notification link_url will only be valid for life of token)"
                )
                login_data = {
                    "user": user.get_secret_value(),
                    "pass": pw.get_secret_value(),
                }
                url = f"{api.api_url}/host/login.json"
                try:
                    login_response = requests.post(url, data=login_data)
                    login_response.raise_for_status()
                    login_response_json = login_response.json()
                except Exception as exc:
                    logger.error(
                        f"{lp} Error trying to obtain user: '{user.get_secret_value()}' token for push "
                        f"notifications, token will not be provided"
                    )
                    logger.debug(f"{lp} EXCEPTION>>> {exc}")
                else:
                    push_auth = f"token={login_response_json.get('access_token')}"
                    logger.debug(f"{lp} token retrieved!")

        else:
            logger.warning(f"{lp} pw not set while user is set")
            # need password with username!
            push_auth = f""

    else:
        logger.debug(f"{lp} link_url NO USER set, using creds from ZM API")
        push_auth = f""

    if not push_auth:
        if api.access_token:
            # Uses the zm_user and zm_password that ZMES uses if push_user and push_pass not set
            logger.warning(
                f"{lp} there does not seem to be a user and/or pass set using credentials from ZM API"
            )
            payload = {
                "user": api.username.get_secret_value(),
                "pass": api.password.get_secret_value(),
            }
            push_auth = urlencode(payload, quote_via=quote_plus)
    return push_auth
