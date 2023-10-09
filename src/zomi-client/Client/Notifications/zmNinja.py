import hashlib
import json
import logging
import os
import pickle
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List

from pydantic import BaseModel, Field, field_validator, SecretStr, AnyUrl, IPvAnyAddress

from ..main import get_global_config
from ...Shared.configs import GlobalConfig
from ..Log import CLIENT_LOGGER_NAME

logger = logging.getLogger(CLIENT_LOGGER_NAME)
g: Optional[GlobalConfig] = None


class TokensStructure(BaseModel):
    class TokenData(BaseModel):
        class Invocations(BaseModel):
            at: Optional[int] = None
            count: Optional[int] = None

        pushstate: Optional[bool] = None
        invocations: Invocations = Field(default_factory=Invocations)
        appversion: Optional[str] = None
        intlist: Optional[List[int]] = None
        monlist: Optional[List[int]] = None
        platform: Optional[str] = None

        """
        test_data = {
            "tokens": {
                "<JWT TOKEN DATA>": {
                    "pushstate": "enabled",
                    "invocations": {
                        "at": "1",
                        "count": "2"
                    },
                    "appversion": "1.6.009",
                    "intlist": "0,0",
                    "monlist": "1,2",
                    "platform": "android"
                }
            }
        }
        """

        @field_validator("intlist", "monlist", mode="before")
        @classmethod
        def convert_to_list(cls, v):
            if v:
                if isinstance(v, str):
                    v = [int(i) for i in v.split(",")]

            return v

        @field_validator("pushstate", mode="before")
        @classmethod
        def set_pushstate(cls, v):
            affirmative = ("enabled", "activated")
            negative = ("disabled", "deactivated")

            if v:
                if isinstance(v, str):
                    v = v.casefold()
                    if v in affirmative:
                        v = True
                    elif v in negative:
                        v = False
            return v

    tokens: Dict[str, TokenData]


def _check_same_month(month_: str, set_month_: str):
    return month_ != set_month_


class ZMNinja:
    """Send an FCM push notification to the zmninja App (native push notification)"""
    fcmv1 = namedtuple("fcmv1", "enabled key url")
    default_fcm_per_month: int = 8000
    default_fcm_v1_key: str = (
        "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJnZW5lcmF0b3IiOiJiYXVkbmVvIiwiaWF0IjoxNjU2OTg5MzYzL"
        "CJjbGllbnQiOiJ6bW5pbmphIn0.rqroX7HCWLxplPNxb421exXusW4_EytK-XP3_rleXvc"
    )
    legacy_fcm_key: str = (
        "key=AAAAVH0b7-U:APA91bHL45N58IMbfXh7ACBlsoldrS-qDGIVr9XT-zdg1O0af5TJHYmxbRH-x9gOWs_Clfp1Z"
        "dump-mE9GqIHMYotKCvOaXOgiELzJn-oUl2tmL2k3hRJHGCOP-UdfGSjZE_eGJinb3O"
    )

    default_fcm_v1_url: str = (
        "https://us-central1-zmninja-notification.cloudfunctions.net/zmninja2gcm"
    )
    tokens_file: Path

    def get_month(self, val_: Union[str, int]) -> Union[str, int]:
        """Get the month from the int_to_month dict"""
        # Perl localtime[4] corresponds to a month between 0-11
        int_to_month = {
            0: "January",
            1: "February",
            2: "March",
            3: "April",
            4: "May",
            5: "June",
            6: "July",
            7: "August",
            8: "September",
            9: "October",
            10: "November",
            11: "December",
            # reversed
            "January": 0,
            "February": 1,
            "March": 2,
            "April": 3,
            "May": 4,
            "June": 5,
            "July": 6,
            "August": 7,
            "September": 8,
            "October": 9,
            "November": 10,
            "December": 11,
        }
        if val_ and isinstance(val_, (str, int)):
            if val_ in int_to_month:
                return int_to_month[val_]

    def get_tokens(self) -> Optional[Dict[str, TokensStructure.TokenData]]:
        """Get the tokens from the tokens file. JSON format"""
        lp: str = "fcm::read tokens::"
        if (
                self.tokens_file.exists()
                and self.tokens_file.is_file()
                and os.access(self.tokens_file, os.R_OK)
        ):
            try:
                fcm_tokens = TokensStructure(**json.loads(self.tokens_file.read_text()))
            except Exception as fcm_load_exc:
                logger.error(
                    f"{lp} failed to load tokens.txt into valid JSON: {fcm_load_exc}"
                )
            else:
                return fcm_tokens.tokens
        return None

    @staticmethod
    def test_data():
        test_data = {
            "tokens": {
                "<JWT TOKEN DATA>": {
                    "pushstate": "enabled",
                    "invocations": {"at": "1", "count": "2"},
                    "appversion": "1.6.009",
                    "intlist": "0,0",
                    "monlist": "1,2",
                    "platform": "android",
                }
            }
        }
        return TokensStructure(**test_data)

    def __init__(self):
        lp: str = "zmninja::init::"
        global g
        g = get_global_config()
        self.noti_cfg = g.config.notifications
        self.config = self.noti_cfg.zmninja
        self.fcmv1.enabled = self.config.fcm.v1.enabled
        self.fcmv1.key = self.config.fcm.v1.key or self.default_fcm_v1_key
        self.fcmv1.url = self.config.fcm.v1.url or self.default_fcm_v1_url
        self.tokens_used = []
        self.event_cause = g.event_cause
        self.tokens_file = self.config.fcm.token_file
        self.app_version: str = ""
        self.fcm_tokens: Dict[str, TokensStructure.TokenData] = self.get_tokens()

    def send(self):
        lp: str = "zmninja::send::"
        fcm_tokens = self.fcm_tokens
        if not fcm_tokens:
            logger.warning(f"{lp} no tokens found, cant send zmNinja notification")
            return

        logger.debug(f"{lp} parsing tokens from '{self.tokens_file}'")
        send_fcm: bool = False
        curr_month = datetime.now().strftime("%B")
        total_sent: int = 0
        i: int = 0
        if fcm_tokens and (num_tkns := len(fcm_tokens)):
            for token, token_data in fcm_tokens.items():
                i += 1
                logger.debug(
                    f"{lp}DEBUG>>> STARTING TOKEN LOOP ({i}/{num_tkns}) with '{token}' <<<DEBUG"
                )
                if not token:
                    logger.warning(f"{lp} token is empty, skipping")
                    continue

                if token not in self.tokens_used:
                    monlist = token_data.monlist
                    intlist = token_data.intlist
                    pushstate = token_data.pushstate
                    fcm_month = self.get_month(token_data.invocations.at)
                    total_sent = token_data.invocations.count
                    platform = token_data.platform
                    self.app_version = token_data.appversion
                    fcm_pkl_path = (
                        g.config.system.variable_data_path
                        / f"zmninja-tkn_{token}.pkl"
                    )
                    if pushstate:
                        # pushstate is enabled, now check if the monitor is in the monlist
                        if g.mid in monlist:
                            # check the intlist for 'cool down'

                            for mid, cooldown in zip(monlist, intlist):
                                if g.mid != mid:
                                    continue
                                if cooldown == 0:
                                    # cool down is disabled, check if we are over the count for this token
                                    if _check_same_month(curr_month, fcm_month):
                                        logger.info(
                                            f"{lp} resetting notification count as month has changed "
                                            f"from {fcm_month} to {curr_month}"
                                        )
                                        total_sent = 0
                                        fcm_month = curr_month
                                        send_fcm = True

                                else:
                                    logger.debug(
                                        f"{lp} token has a cooldown of {cooldown}, checking..."
                                    )
                                    # cool down is enabled, read pickled data and compare datetimes
                                    fcm_pkl: Optional[float] = None
                                    if fcm_pkl_path.exists() and os.access(
                                        fcm_pkl_path, os.R_OK
                                    ):
                                        with fcm_pkl_path.open("rb") as f:
                                            fcm_pkl = pickle.load(f)
                                    if fcm_pkl:
                                        cooldown_ = time.time() - fcm_pkl
                                        if cooldown_ > cooldown:
                                            logger.debug(
                                                f"{lp} token has exceeded the cooldown wait "
                                                f"({cooldown}), sending FCM - ELAPSED: {cooldown_}"
                                            )
                                            # cool down has expired,
                                            send_fcm = True
                                            continue
                                        else:
                                            logger.debug(
                                                f"{lp} token has not exceeded the cooldown of "
                                                f"{cooldown}, not sending FCM - ELAPSED: {cooldown_}"
                                            )
                            if send_fcm:
                                if self._check_invocations(total_sent):
                                    # todo write the data out to tokens.txt
                                    total_sent += 1
                                    self.send_fcm(
                                        token=token,
                                        platform=platform,
                                        pkl_path=fcm_pkl_path,
                                    )
                                else:
                                    logger.error(
                                        f"{lp} token has exceeded the max FCM invocations per "
                                        f"month ({self.default_fcm_per_month}, not sending FCM"
                                    )

                                send_fcm = False

                        else:
                            logger.debug(
                                f"{lp} monitor {g.mid} is not enabled for this token"
                            )
                    else:
                        logger.info(
                            f"{lp} token pushstate is disabled, not sending notification"
                        )

                else:
                    logger.debug(f"{lp} token {token} has already been sent a notification")
        else:
            logger.error(f"{lp} no zmNinja tokens found, not sending FCM")
        self.write_tokens()

    def send_fcm(
        self, token: str, platform: str, pkl_path: Path
    ) -> None:
        """
        Send FCM message to specified token
        :param pkl_path: Path to pickle file
        :param platform: android or ios
        :param token: The token to send the notification to
        :return: None
        """
        lp: str = "zmninja::send fcm::"
        logger.info(f"{lp} sending FCM to token {token[:-30]}")
        key_: str = ""
        url: str = ""
        title: str = f"{g.mon_name} Alarm ({g.eid})"
        zm_user: SecretStr = g.config.zoneminder.user
        zm_pass: SecretStr = g.config.zoneminder.password
        date_fmt: str = self.config.fcm.date_fmt

        fcm_log_message_id = self.config.fcm.log_message_id
        fcm_log_ = self.config.fcm.log_raw_message
        use_fcmv1: bool = self.config.fcm.v1.enabled
        replace_push_messages = self.config.fcm.replace_messages
        android_ttl = self.config.fcm.android_ttl
        android_priority = self.config.fcm.android_priority
        image_auth: str = ""

        image_url = f"{g.api.portal_base_url}/index.php?view=image&eid={g.eid}&fid=objdetect&width=600"
        body: str = f"{g.event_cause} started at {datetime.now().strftime(date_fmt)}"

        if zm_user and zm_pass:
            from urllib.parse import urlencode, quote_plus

            payload = {'username': zm_user.get_secret_value(), 'password': zm_pass.get_secret_value()}
            image_auth = urlencode(payload, quote_via=quote_plus)


        image_url = f"{image_url}{image_auth}"

        """
        my $android_message = {
            to           => $obj->{token},
            notification => {
              title              => $title,
              android_channel_id => 'zmninja',
              icon               => 'ic_stat_notification',
              body               => $body,
              sound              => 'default',
              badge              => $badge,
            },
            data => {
              title       => $title,
              message     => $body,
              style       => 'inbox',
              myMessageId => $notId,
              icon        => 'ic_stat_notification',
              mid         => $mid,
              eid         => $eid,
              badge       => $obj->{badge},
              priority    => 1
            }
          };
          """
        if not use_fcmv1:
            # Legacy FCM
            _key = self.legacy_fcm_key
            logger.debug(
                f"{lp}DEBUG>>> FCM: Using LEGACY FCM (fcm:v1:enabled = {use_fcmv1})"
            )
            url = "https://fcm.googleapis.com/fcm/send"

            from random import randint

            message = {
                "to": token,
                "notification": {
                    "title": title,
                    "android_channel_id": "zmninja",
                    "icon": "ic_stat_notification",
                    "body": body,
                    "sound": "default",
                    "badge": 1,
                    "image": image_url,
                },
                "data": {
                    "title": title,
                    "message": body,
                    "style": "picture",
                    "picture": image_url,
                    "summaryText": f"Detection results",
                    "myMessageId": randint(1, 654321),
                    "icon": "ic_stat_notification",
                    "mid": g.mid,
                    "eid": g.eid,
                    "badge": 1,
                    "priority": 1,
                    "channel": "zmninja",
                },
            }

        else:
            # FCM v1
            url = self.default_fcm_v1_url
            _key = self.default_fcm_v1_key

            message = {
                "token": token,
                "title": title,
                "body": body,
                # 'image_url': self.image_url,
                "sound": "default",
                # 'badge': int(self.badge),
                "log_message_id": fcm_log_message_id,
                "data": {"mid": g.mid, "eid": g.eid, "notification_foreground": "true"},
                "image_url": image_url,
            }
            if platform == "android":
                message["android"] = {
                    "icon": "ic_stat_notification",
                    "priority": android_priority,
                }
                if android_ttl is not None:
                    message["android"]["ttl"] = android_ttl
                if replace_push_messages:
                    message["android"]["tag"] = "zmninjapush"
                if self.app_version and self.app_version != "unknown":
                    logger.debug(f"{lp} setting channel to zmninja")
                    message["android"]["channel"] = "zmninja"
                else:
                    logger.debug(f"{lp} legacy client, NOT setting channel to zmninja")

        if platform == "ios":
            message["ios"] = {
                "thread_id": "zmninja_alarm",
                "headers": {
                    "apns-priority": "10",
                    "apns-push-type": "alert",
                    # 'apns-expiration': '0'
                },
            }
            if replace_push_messages:
                message["ios"]["headers"]["apns-collapse-id"] = "zmninjapush"
        logger.debug(f"DEBUG>>>> IMAGE URL = {image_url} <<<<DEBUG")

        if fcm_log_:
            message["log_raw_message"] = "yes"
            logger.debug(
                f"{lp} The server cloud function at {url} will log your full message. "
                f"Please ONLY USE THIS FOR DEBUGGING and turn off later"
            )
        # send the message with header auth
        headers = {
            "content-type": "application/json",
            "Authorization": _key,
        }

        from requests import post
        logger.debug(f"{lp}DEBUG>>> FCM: URL = {url}")
        logger.debug(f"{lp}DEBUG>>> FCM: KEY = {_key}")
        logger.debug(f"{lp}DEBUG>>> FCM: MESSAGE = {message}")
        logger.debug(f"{lp}DEBUG>>> FCM: HEADERS = {headers}")

        response_ = post(url, data=json.dumps(message), headers=headers)
        if response_ and response_.status_code == 200:
            logger.debug(
                f"{lp} sent successfully to token - response message: {response_.text}"
            )
            self.tokens_used.append(token)
            if pkl_path:
                logger.debug(f"{lp} serializing timestamp to {pkl_path}")
                try:
                    pkl_path.touch(exist_ok=True, mode=0o640)
                    # write_bytes as pickle is binary data
                    pkl_path.write_bytes(pickle.dumps(time.time()))
                except Exception as e:
                    logger.error(
                        f"{lp} failed to serialize timestamp object to {pkl_path}"
                    )

        elif response_:
            logger.error(
                f"{lp} FCM failed to send to token with error code: {response_.status_code} - "
                f"response message: {response_.text}"
            )
            if (
                response_.text.find("not a valid FCM") > -1
                or response_.text.find("entity was not found") > -1
            ):
                # todo remove the token from the file

                logger.warning(
                    f"{lp} FCM token is invalid, REMOVING..."
                )
                self.fcm_tokens.pop(token)
        else:
            logger.error(f"{lp} FCM failed to send to token")
            logger.debug(f"DEBUG FCM {response_ = } - {response_.text}")

    def write_tokens(self):
        logger.debug(f"DEBUG>>> {self.fcm_tokens = }")
        if self.fcm_tokens:
            self.tokens_file.write_bytes(pickle.dumps(self.fcm_tokens))

    def _check_invocations(self, count: int) -> bool:
        """Check if we have exceeded the max FCM invocations per month"""
        # "invocations": {"at":1, "count":0}
        if count < self.default_fcm_per_month:
            return True
        return False
