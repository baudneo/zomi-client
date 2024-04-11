import logging
from typing import Optional, Union, List
from pathlib import Path
import json
from datetime import datetime, timedelta

from ..Log import CLIENT_LOGGER_NAME
from ..Models.config import (
    MLNotificationSettings,
    CoolDownSettings,
    OverRideCoolDownSettings,
)

logger = logging.getLogger(CLIENT_LOGGER_NAME)


class CoolDownBase:
    _cooldown_str = ".cooldown_m{mid}.json"
    data_dir: Path
    lp: str = "CoolDown:"
    config: Union[
        MLNotificationSettings.GotifyNotificationSettings,
        MLNotificationSettings.PushoverNotificationSettings,
        MLNotificationSettings.ZMNinjaNotificationSettings,
        MLNotificationSettings.MQTTNotificationSettings,
        MLNotificationSettings.ShellScriptNotificationSettings,
    ]
    cooldown: Union[CoolDownSettings, OverRideCoolDownSettings]

    def __init__(self):
        if self.config.cooldown:
            self.cooldown = self.config.cooldown
        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def write_cooldown(self, mid: int):
        cooldown_file: Path = self.data_dir / self._cooldown_str.format(mid=mid)
        cooldown_file.expanduser().resolve()
        try:
            with cooldown_file.open("w") as f:
                json.dump({"last_sent": datetime.timestamp(datetime.now())}, f)
        except Exception as e:
            logger.error(
                f"{self.lp} Failed to write file: {cooldown_file} -- {e}"
            )
        else:
            logger.debug(f"{self.lp} File written: {cooldown_file}")

    def check_cooldown(self, mid: int):
        # JSON: {'last_sent': 1685383194.070348 }
        linked: Optional[List] = None
        if getattr(self.cooldown, "linked", False):
            linked = self.cooldown.linked
        ids = []
        if linked:
            ids = [int(str(x).strip()) for x in linked]
        ids.append(mid)
        for _mid in ids:
            cooldown_file: Path = self.data_dir / self._cooldown_str.format(mid=_mid)
            cooldown_file.expanduser().resolve()
            if cooldown_file.exists():
                try:
                    with cooldown_file.open("r") as f:
                        data = json.load(f)
                except Exception as e:
                    logger.error(
                        f"{self.lp} Failed to read file: {cooldown_file} -- {e}"
                    )
                else:
                    if data:
                        last_sent = data.get("last_sent")
                        if last_sent:
                            last_sent_ts = datetime.fromtimestamp(last_sent)
                            cooldown = timedelta(seconds=self.cooldown.seconds)
                            # check if cooldown has passed
                            if (datetime.now() - last_sent_ts) < cooldown:
                                logger.warning(
                                    f"{self.lp} Monitor {mid} cooldown has not passed yet, skipping notification..."
                                )
                                return False
                            else:
                                logger.debug(
                                    f"{self.lp} Monitor {mid} cooldown has passed, sending a notification"
                                )
            else:
                logger.warning(
                    f"{self.lp} file does not exist: {cooldown_file.as_posix()}"
                )
        return True
