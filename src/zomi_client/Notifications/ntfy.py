from __future__ import annotations

import json
import logging
import requests
from typing import Optional, TYPE_CHECKING

from ..Log import CLIENT_LOGGER_NAME
from ..main import get_global_config
from ..Notifications import CoolDownBase

if TYPE_CHECKING:
    from ..Models.config import GlobalConfig

logger = logging.getLogger(CLIENT_LOGGER_NAME)

g: Optional[GlobalConfig] = None
LP: str = 'ntfy:'

class NtfyNotification(CoolDownBase):
    def __init__(self):
        global g

        g = get_global_config()
        self.config = g.config.notifications.ntfy
        self.url = self.config.url
        self.topic = self.config.topic
        self.priority = self.config.priority
        self.tags = self.config.tags
        self.token = self.config.token


    def prepare_payload(self, message: str):
        url_opts = g.config.notifications.url_opts
        _mode = url_opts.mode
        _scale = url_opts.scale
        _max_fps = url_opts.max_fps
        _buffer = url_opts.buffer
        _replay = url_opts.replay
        portal = self.config.portal
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token.get_secret_value()}"
        clickable_link = self.config.clickable_link
        link_user = g.config.notifications.link_user
        link_pass = g.config.notifications.link_pass
        image_auth = ""
        event_auth = ""
        event_url = ""
        zm_user = g.config.zoneminder.api.user
        zm_pass = g.config.zoneminder.api.password
        from urllib.parse import urlencode, quote_plus

        if zm_user and zm_pass:
            payload = {
                "username": zm_user.get_secret_value(),
                "password": zm_pass.get_secret_value(),
            }
            image_auth = urlencode(payload, quote_via=quote_plus)
        image_url: str = (
            f"{portal}/index.php?view=image&eid={g.eid}&fid="
            f"objdetect&{image_auth}"
        )

        payload = {
            "topic": self.topic,
            "message": message,
            "title": "Detection Alert",
            # "tags": tags,
            # "priority": priority,
            "attach": image_url,
            "filename": "objdet.jpg",
            # "click": "https://homecamera.lan/xasds1h2xsSsa/",
            # "actions": [{"action": "view", "label": "Admin panel", "url": "https://filesrv.lan/admin"}]
            "actions": [],
        }
        actions = payload["actions"]
        if clickable_link:
            if link_user and link_pass:
                eauth_payload = {
                    "user": link_user.get_secret_value(),
                    "pass": link_pass.get_secret_value(),
                }
                event_auth = urlencode(eauth_payload, quote_via=quote_plus)
            event_url = (
                f"{portal}/cgi-bin/nph-zms?mode={_mode}&scale="
                f"{_scale}&maxfps={_max_fps}&buffer={_buffer}&replay={_replay}&"
                f"monitor={g.mid}&event={g.eid}&{event_auth}"
            )
            actions.append({"action": "view", "label": "View Event", "url": event_url})

        if self.tags:
            payload["tags"] = self.tags
        if self.priority:
            payload["priority"] = self.priority

        if headers:
            payload["headers"] = headers

        return payload

    async def async_send(self, message: str):
        if not self.config.enabled:
            logger.debug("Ntfy notifications are disabled.")
            return
        session = g.api.async_session
        if session is not None:
            headers = {}
            payload = self.prepare_payload(message)
            if 'headers' in payload:
                headers = payload.pop('headers')
            try:
                async with session.post(
                    self.url, data=json.dumps(payload), headers=headers
                ) as response:
                    response.raise_for_status()
            except Exception as e:
                logger.error(f"Failed to send ntfy notification: {e}", exc_info=True)
            else:
                logger.debug("Ntfy notification sent successfully.")

    def send(self, message: str):
        if not self.config.enabled:
            logger.debug("Ntfy notifications are disabled.")
            return
        headers = {}
        payload = self.prepare_payload(message)
        if 'headers' in payload:
            headers = payload.pop('headers')
            logger.debug(f"Headers: {headers}")
        logger.debug(f"Payload: {payload}")
        logger.debug(f"{LP} Ntfy config: {self.config}")
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            logger.debug("Ntfy notification sent successfully.")
        except requests.RequestException as e:
            logger.error(f"Failed to send ntfy notification: {e}")