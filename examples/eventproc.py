#!/opt/zomi/client/venv/bin/python3

from __future__ import annotations

import logging.handlers
import sys
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import uvloop
import asyncio

try:
    import zomi_client
except ImportError:
    zomi_client = None
    import_err_msg = (
        "ZoMi Client library not installed! Please install zomi-client: "
        "https://github.com/baudneo/zomi-client"
    )
    print(import_err_msg)
    raise ImportError(import_err_msg)

from zomi_client.Models.validators import str2path
from zomi_client.Models.config import ClientEnvVars, GlobalConfig
from zomi_client.main import (
    parse_client_config_file,
    create_global_config,
    create_logs,
)

if TYPE_CHECKING:
    from zomi_client.main import ZMClient

__doc__ = """
A script that uses ZoneMinders EventStartCommand/EventEndCommand mechanism to run 
object detection on an event.
"""
# Setup basic console logging (hook into library logging)
logger: logging.Logger = create_logs()
zm_client: Optional[ZMClient] = None


def _parse_cli():
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="eventproc.py", description=__doc__)
    parser.add_argument(
        "--event-start",
        help="This is the start of an event",
        action="store_true",
        dest="event_start",
        default=True,
    )
    parser.add_argument(
        "--event-end",
        help="This is the end of an event",
        action="store_false",
        dest="event_start",
    )

    parser.add_argument("--config", "-C", help="Config file to use", type=Path)
    parser.add_argument(
        "--event-mode",
        "-E",
        action="store_true",
        dest="event",
        help="Run in event mode (triggered by a ZM event)",
    )
    parser.add_argument(
        "--event-id",
        "--eid",
        "-e",
        help="Event ID to process (Required for --event/-E)",
        type=int,
        dest="eid",
        default=0,
    )

    parser.add_argument(
        "--monitor-id",
        "--mid",
        "-m",
        help="Monitor ID to process",
        type=int,
        dest="mid",
        default=0,
    )
    parser.add_argument(
        "--debug", "-D", help="Enable debug logging", action="store_true", dest="debug"
    )
    parser.add_argument(
        "--live",
        "-L",
        help="This is a live event",
        action="store_true",
        dest="live",

    )

    args = parser.parse_args()
    # logger.debug(f"CLI Args: {args}")
    return args


async def main():
    global g, zm_client
    _mode = ""
    _start = time.time()
    # Do all the config stuff and setup logging
    lp = f"event-start:" if args.event_start else f"event-end:"
    eid = args.eid
    mid = args.mid
    event_start = args.event_start
    cfg_file: Optional[Path] = None
    if args.event:
        _mode = "event"
        logger.debug(f"{lp} Running in event mode")
        if eid == 0:
            logger.error(f"{lp} Event ID is required for event mode")
            sys.exit(1)

    if "config" in args and args.config:
        cfg_file = args.config
    else:
        logger.warning(
            f"No config file supplied, checking ENV: {g.Environment.client_conf_file}"
        )
        if g.Environment.client_conf_file:
            cfg_file = g.Environment.client_conf_file
    if cfg_file:
        cfg_file = str2path(cfg_file)
    assert cfg_file, "No config file supplied via CLI or ENV"
    g.config_file = cfg_file
    logger.info(
        f"{lp} Event: {args.event} [Event ID: {eid}]"
        f"{' || Monitor ID: ' if mid else ''}{mid if mid else ''} || "
        f"Config File: {cfg_file}"
    )
    g.config = parse_client_config_file(cfg_file)
    zm_client = zomi_client.main.ZMClient(global_config=g)
    _end_init = time.time()
    __event_modes = ["event", ""]
    if _mode in __event_modes:
        # set live or past event
        zm_client.is_live_event(args.live)
        return await zm_client.detect(eid=eid, mid=g.mid)
    else:
        raise ValueError(f"Unknown mode: {_mode}")


if __name__ == "__main__":
    filename = Path(__file__).stem
    args = _parse_cli()
    logger.debug(f"Starting {filename}...")
    ENV_VARS = ClientEnvVars()
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
    _start = time.time()
    logger.debug(f"ENV VARS: {ENV_VARS}")
    g: GlobalConfig = create_global_config()
    g.Environment = ENV_VARS
    detections = None
    try:
        detections = loop.run_until_complete(main())
    except Exception as e:
        logger.error(f"eventproc: Error in main(): {e}", exc_info=True)
        from zomi_client import Log
        for handler in logger.handlers:
            if isinstance(handler, Log.BufferedLogHandler):
                # should only print out if there is no file logging going on
                handler.flush2()
    # Allow 250ms for aiohttp SSL session context to close properly
    # loop.run_until_complete(asyncio.sleep(0.25))
    if detections is not None:
        logger.debug(f"DETECTIONS: {detections}")

    final_msg = f"perf:FINAL: Event processing took {time.time() - _start:.5f} seconds"
    if not loop.is_closed():
        loop.run_until_complete(zm_client.clean_up())
        logger.info(final_msg)
        loop.close()
    else:
        logger.info(final_msg)
