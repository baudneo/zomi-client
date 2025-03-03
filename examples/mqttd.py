#!/opt/zomi/client/venv/bin/python3

"""Wrapper script to run zomi-client as a MQTT client daemon"""

from __future__ import annotations

import asyncio
import logging.handlers
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import uvloop

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
A script that subscribes to ZoneMinders MQTT motion events to run 
machine learning models on an event.
"""
# Setup basic console logging (hook into library logging)
logger: logging.Logger = create_logs()
zm_client: Optional[ZMClient] = None
filename = Path(__file__).stem
LP = filename.split('.')[0] + ':'

def _parse_cli():
    from argparse import ArgumentParser

    parser = ArgumentParser(prog=filename, description=__doc__)

    parser.add_argument("--config", "-C", help="Config file to use", type=Path)
    parser.add_argument(
        "--debug", "-D", help="Enable debug logging", action="store_true", dest="debug"
    )
    args = parser.parse_args()
    # logger.debug(f"CLI Args: {args}")
    return args


if __name__ == "__main__":
    _start = time.time()
    args = _parse_cli()
    logger.debug(f"{LP} Starting...")
    # logger.debug(f"DBG>>> env: {os.environ.items()}")
    ENV_VARS = ClientEnvVars()
    uvloop.install()
    loop = asyncio.get_event_loop()
    logger.debug(f"{LP} ENV VARS: {ENV_VARS}")
    g: GlobalConfig = create_global_config()
    g.Environment = ENV_VARS
    cfg_file: Optional[Path] = None

    if "config" in args and args.config:
        cfg_file = args.config
    else:
        logger.warning(
            f"{LP} No config file supplied via CLI, checking ENV: {g.Environment.client_conf_file}"
        )
        if g.Environment.client_conf_file:
            cfg_file = g.Environment.client_conf_file
    if cfg_file:
        cfg_file = str2path(cfg_file)
    assert cfg_file, f"{LP} No config file supplied via CLI or ENV"
    g.config_file = cfg_file

    g.config = parse_client_config_file(cfg_file)
    if not g.config.mqtt:
        raise ValueError("No MQTT config found in config file!")
    zm_client = zomi_client.main.ZMClient(global_config=g)
    try:
        logger.debug("STARTING async mqtt_start()")
        loop.run_until_complete(zm_client.mqtt_start())
    except Exception as e:
        logger.error(f"{LP} Error in main(): {e}", exc_info=True)
        # from zomi_client import Log
        # for handler in logger.handlers:
        #     if isinstance(handler, Log.BufferedLogHandler):
        #         # should only print out if there is no file logging going on
        #         handler.flush2()

    logger.info(f"perf:{LP}FINAL:  Lifetime -> {time.time() - _start:.5f} seconds")
    if not loop.is_closed():
        loop.close()
