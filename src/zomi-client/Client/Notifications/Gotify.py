from __future__ import annotations
import logging
from typing import Optional, TYPE_CHECKING

import requests

from ..Log import CLIENT_LOGGER_NAME
from ..Notifications import CoolDownBase
from ..main import get_global_config
if TYPE_CHECKING:
    from ...Shared.configs import GlobalConfig

logger = logging.getLogger(CLIENT_LOGGER_NAME)
g: Optional[GlobalConfig] = None
LP: str = "Gotify:"


class Gotify(CoolDownBase):
    _data_dir_str = "push/gotify"

    def __init__(self):
        global g
        g = get_global_config()
        self.lp = "Gotify:"
        self.config = g.config.notifications.gotify
        self.data_dir = g.config.system.variable_data_path / self._data_dir_str
        self.title = f"({g.eid}) {g.mon_name}->{g.event_cause}"
        self.message = ""
        super().__init__()

    def send(self, pred_out: str):
        if not self.check_cooldown(g.mid):
            return
        lp = "gotify::send::"
        url_opts = self.config.url_opts
        _mode = url_opts.mode
        _scale = url_opts.scale
        _max_fps = url_opts.max_fps
        _buffer = url_opts.buffer
        _replay = url_opts.replay
        portal = self.config.portal or g.api.portal_base_url
        host = self.config.host
        token = self.config.token
        link_url = self.config.clickable_link
        link_user = self.config.link_user
        link_pass = self.config.link_pass

        image_auth = ""
        event_auth = ""
        event_url = ""

        zm_user = g.config.zoneminder.api.user
        zm_pass = g.config.zoneminder.api.password
        _link_url = ""
        _embedded_event = ""
        from urllib.parse import urlencode, quote_plus

        if zm_user and zm_pass:
            payload = {
                "username": zm_user.get_secret_value(),
                "password": zm_pass.get_secret_value(),
            }
            image_auth = urlencode(payload, quote_via=quote_plus)
        if link_url:
            if link_user and link_pass:
                payload = {
                    "user": link_user.get_secret_value(),
                    "pass": link_pass.get_secret_value(),
                }
                event_auth = urlencode(payload, quote_via=quote_plus)
            event_url = (
                f"{portal}/cgi-bin/nph-zms?mode={_mode}&scale="
                f"{_scale}&maxfps={_max_fps}&buffer={_buffer}&replay={_replay}&"
                f"monitor={g.mid}&event={g.eid}&{event_auth}"
            )
            _link_url = f"\n[View event in browser]({event_url})"
            # _embedded_event = f"![Embedded event video]({event_url})"
        try:
            # goti_image_url: str = f"{g.api.portal_url}/index.php?view=image&eid={g.eid}&fid=objdetect&{push_zm_tkn}"
            image_url: str = (
                f"{portal}/index.php?view=image&eid={g.eid}&fid="
                f"objdetect&{image_auth}"
            )
            # logger.debug(f"{image_url = } -- | -- {event_url = }")
            test_ = "https://placekitten.com/400/200"
            markdown_formatted_msg: str = (
                f"{pred_out}\n{_link_url}"
                f"![detection.jpeg]({image_url})\n"
                f"{_embedded_event}"
            )
            # \n![Embedded event video for gotify web app]({goti_event_url})

            data = {
                "title": self.title,
                "message": markdown_formatted_msg,
                "priority": 100,
                "extras": {
                    "client::display": {"contentType": "text/markdown"},
                    "client::notification": {
                        "bigImageUrl": f"{image_url}",
                        # "click": {
                        #     "url": f"{goti_event_url}",
                        # },
                    },
                },
            }
            goti_url = f"{host}message?token={token}"
            resp = requests.post(goti_url, json=data)
            resp.raise_for_status()
        except Exception as custom_push_exc:
            logger.error(f"{lp} ERROR while sending Gotify notification ")
            logger.debug(f"{lp} EXCEPTION>>> {custom_push_exc}", exc_info=True)
        else:
            if resp:
                if resp.status_code == 200:
                    logger.debug(f"{lp} Gotify returned SUCCESS")
                    self.write_cooldown(g.mid)
                elif resp.status_code != 200:
                    logger.debug(
                        f"{lp} Gotify FAILURE STATUS_CODE: {resp.status_code} -> {resp.json()}"
                    )

            else:
                logger.debug(f"{lp} Gotify FAILURE NO RESPONSE")
                logger.warning(f"{lp} Gotify failure no response - Possible 401 unauthorized, is the token correct?")

