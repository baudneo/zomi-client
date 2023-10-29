from __future__ import annotations

import logging
import ssl
from time import perf_counter
from typing import Union, Optional, TYPE_CHECKING, Any, Dict

import numpy as np
import paho.mqtt.client as paho_client

from ..Log import CLIENT_LOGGER_NAME
from ..Models.config import MLNotificationSettings
from ..Notifications import CoolDownBase
from ..main import get_global_config

if TYPE_CHECKING:
    from ..Models.config import GlobalConfig

g: Optional[GlobalConfig] = None
logger = logging.getLogger(CLIENT_LOGGER_NAME)
LP: str = "MQTT:"


class MQTT(CoolDownBase):
    """Create an MQTT object to publish data"""

    _data_dir_str: str = "push/mqtt"
    conn_timeout: float = 5.0
    client: Optional[paho_client.Client] = None
    _connected: bool = False
    conn_time: Optional[float] = None
    ssl_cert: int = ssl.CERT_REQUIRED
    client_id: Optional[str] = None
    config: MLNotificationSettings.MQTTNotificationSettings
    _image: Optional[Union[str, bytes, np.ndarray]] = None
    sanitize: bool = False
    sanitize_str: Optional[str] = None

    def __init__(self):
        global g
        g = get_global_config()
        self.config = g.config.notifications.mqtt
        self.data_dir = (
            (g.config.system.variable_data_path / self._data_dir_str)
            .expanduser()
            .resolve()
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.id_file = self.data_dir / "client_id"
        if self.id_file.is_file():
            self.client_id = self.id_file.read_text()
        self.id_file.touch(exist_ok=True, mode=0o660)
        if not self.client_id:
            import uuid

            self.client_id = uuid.uuid4().__str__()
        self.id_file.write_text(self.client_id)
        self.client_id = f"zomi-client_{self.client_id}"

        self.sanitize = g.config.logging.sanitize.enabled
        self.sanitize_str = g.config.logging.sanitize.replacement_str

        if not self.config.port:
            self.config.port = 1883
        if self.config.tls_secure is False:
            self.ssl_cert = ssl.CERT_NONE
        super().__init__()

    @staticmethod
    def _on_log(client: paho_client.Client, userdata, level, buf):
        logger.debug(f"{LP}paho_log: {buf}")

    @staticmethod
    def _on_publish(client: paho_client.Client, userdata, mid):
        logger.debug(f"{LP}on_publish: message_id: {mid = }")

    def _on_connect(self, client: paho_client.Client, userdata, flags, rc):
        lp = f"{LP}on_connect: "
        if rc == 0:
            logger.debug(f"{lp} connected to broker with flags-> {flags}")
            self._connected = True
        else:
            logger.debug(f"{lp} connection failed with result code-> {rc}")

    @property
    def connected(self):
        return self._connected

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image: Union[np.ndarray, bytes]):
        if isinstance(image, np.ndarray):
            self._image = self.encode_image(image)
        elif isinstance(image, bytes):
            self._image = image

    def encode_image(self, image: np.ndarray):
        """Prepares an image to be published, tested on jpg and gif so far. Give it an image and
        it then converts it to a byte array or base64 string depending on the config.image.format setting
        """
        import cv2

        lp = f"{LP}encode_image:"
        _type = self.config.image.format
        if not _type:
            _type = "bytes"
        if _type not in ["bytes", "base64"]:
            logger.warning(f"{lp} invalid image format: {_type}")
        else:
            succ, jpg = cv2.imencode(".jpeg", image)
            if not succ:
                logger.warning(f"{lp} failed to encode image to jpeg")
            else:
                if _type == "bytes":
                    logger.debug(f"{lp} encoding image using bytes")
                    image = jpg.tobytes()
                else:
                    import base64

                    logger.debug(f"{lp} encoding image using base64")
                    image = base64.b64encode(jpg.tobytes())
                del jpg
        return image

    def connect(self):
        if not self.config.keep_alive:
            self.config.keep_alive = 60
        else:
            self.config.keep_alive = self.config.keep_alive
        if self.config.qos is None:
            self.config.qos = 0
        if self.config.retain is None:
            self.config.retain = False
        from uuid import uuid4

        uuid = uuid4()
        try:
            ca = self.config.tls_ca.expanduser().resolve()
            tls_key = self.config.tls_key.expanduser().resolve()
            tls_cert = self.config.tls_cert.expanduser().resolve()

            # First check if there is a CA cert, if there is then we are using TLS
            if ca:
                if ca.is_file():
                    logger.debug(
                        f"{LP}connect: TLS CA cert found, checking mTLS cert and key"
                    )
                    if tls_cert and tls_key:
                        if tls_cert.is_file() and tls_key.is_file():
                            logger.debug(
                                f"{LP}connect: TLS cert and key found, assuming mTLS"
                            )
                            # mTLS cert and key are both present
                            self.client_id = f"{self.client_id}mTLS-{uuid}"
                            self.client = paho_client.Client(self.client_id)
                            self.client.tls_set(
                                ca_certs=self.config.tls_ca.expanduser()
                                .resolve()
                                .as_posix(),
                                certfile=self.config.tls_cert.expanduser()
                                .resolve()
                                .as_posix(),
                                keyfile=self.config.tls_key.expanduser()
                                .resolve()
                                .as_posix(),
                                cert_reqs=self.ssl_cert,
                                # tls_version=ssl.PROTOCOL_TLSv1_2
                            )
                            # if self.config.tls_secure is False:
                            #     DON'T verify CN (COMMON NAME) in certificates
                            #     self.client.tls_insecure_set(True)
                            logger.debug(
                                f"{LP}connect: ID: '{self.client_id}' ->  '{self.config.broker if not self.sanitize else self.sanitize_str}:{self.config.port}' trying mTLS "
                                f"({'TLS Secure' if self.config.tls_secure else 'TLS Insecure'}) -> tls_ca: "
                                f"'{self.config.tls_ca}' tls_client_key: '{self.config.tls_key}' tls_client_cert: '{self.config.tls_cert}'"
                            )
                        elif not tls_key.is_file():
                            logger.warning(
                                f"{LP}connect: TLS key not found, cannot use mTLS!"
                            )
                        elif not tls_cert.is_file():
                            logger.warning(
                                f"{LP}connect: TLS cert not found, cannot use mTLS!"
                            )
                    elif tls_cert and not tls_key:
                        logger.warning(
                            f"{LP}connect: TLS cert SUPPLIED but not key, cannot use mTLS!"
                        )

                    elif not tls_cert and tls_key:
                        logger.warning(
                            f"{LP}connect: TLS key SUPPLIED but not cert, cannot use mTLS!"
                        )
                    else:
                        self.client_id = f"{self.client_id}TLS-{uuid}"
                        self.client = paho_client.Client(self.client_id)
                        self.client.tls_set(
                            self.config.tls_ca.expanduser().resolve().as_posix(),
                            cert_reqs=self.ssl_cert,
                        )
                        logger.debug(
                            f"{LP}connect: {self.client_id} -> {self.config.broker if not self.sanitize else self.sanitize_str}:{self.config.port} trying TLS "
                            f"({'TLS Secure' if self.config.tls_secure else 'TLS Insecure'}) -> tls_ca: {self.config.tls_ca}"
                        )
                else:
                    logger.warning(
                        f"{LP}connect: TLS CA cert not found, cannot use TLS!"
                    )

            if not self.client:
                # No tls_ca so we are not using TLS
                self.client_id = f"{self.client_id}noTLS-{uuid}"
                self.client = paho_client.Client(self.client_id)
                logger.debug(
                    f"{LP}connect: ID: {self.client_id} -> "
                    f"{self.config.broker if not self.sanitize else self.sanitize_str}:{self.config.port} "
                    f"{'user: {}'.format(self.config.user) if self.config.user else '<None>'} "
                    f"{'passwd: {}'.format(self.config.pass_.get_secret_value() if not self.sanitize else self.sanitize_str) if self.config.pass_ and self.config.user else 'passwd: <None>'}"
                )

            if self.config.user and self.config.pass_:
                self.client.username_pw_set(
                    self.config.user, password=self.config.pass_.get_secret_value()
                )
            self.client.connect_async(
                self.config.broker,
                port=self.config.port,
                keepalive=self.config.keep_alive,
            )  # connect to broker
            self.client.loop_start()  # start the loop
            self.client.on_connect = self._on_connect  # attach function to callback
            # self.client.on_log = on_log
        except Exception as e:
            logger.error(f"{LP}connect:err_msg-> {e}")

        if not self.client:
            logger.error(
                f"{LP}connect: STRANGE ERROR -> there is no active mqtt object instantiated?! Exiting mqtt routine"
            )
            return
        logger.debug(
            f"{LP}connect: connecting to broker (timeout: {self.conn_timeout})"
        )
        start = perf_counter()
        while not self._connected:
            if (perf_counter() - start) >= self.conn_timeout:
                logger.error(
                    f"{LP}connect: broker: '{self.config.broker if not self.sanitize else self.sanitize_str}' did not reply within '{self.conn_timeout}' seconds"
                )
                return
        else:
            self.conn_time = perf_counter()

    def send(self, *args, **kwargs):
        self.publish(*args, **kwargs)

    def publish(
        self,
        mid: int,
        root_topic: Optional[str] = None,
        fmt_str: Optional[str] = None,
        results: Optional[Dict[str, Any]] = None,
        image: Optional[np.ndarray] = None,
    ):
        if not self._connected:
            logger.error(
                f"{LP}publish: not connected to broker, please connect first, skipping publish..."
            )
            return
        if not fmt_str or not results or image is None:
            logger.warning(f"{LP}publish: no message or image to publish, skipping...")
            return

        if not root_topic:
            if self.config.root_topic:
                root_topic = f"{self.config.root_topic}/mid/{mid}"
            else:
                root_topic = f"zomi/mid/{mid}"

        if fmt_str or results:
            if fmt_str:
                topic = f"{root_topic}/result/formatted"
                message = fmt_str
                logger.debug(f"{LP}publish: topic: '{topic}' data: {message[:255]}")
                self.client.publish(
                    f"{topic}", message, qos=self.config.qos, retain=self.config.retain
                )
            if results:
                import json

                message = json.dumps(results)
                topic = f"{root_topic}/result/json"
                logger.debug(f"{LP}publish: topic: '{topic}' data: {message[:255]}")
                self.client.publish(
                    f"{topic}", message, qos=self.config.qos, retain=self.config.retain
                )
        message = None
        if self.config.image.enabled:
            if image is not None:

                def _get_size(msg: Union[bytes, bytearray, str]) -> str:
                    size_str = ""
                    if hasattr(msg, "__sizeof__"):
                        size = msg.__sizeof__() / 1024 / 1024
                        if size < 1:
                            size = msg.__sizeof__() / 1024
                            if size < 1:
                                size_str = f"{msg.__sizeof__():.2f} B"
                            else:
                                size_str = f"{size:.2f} KB"
                        else:
                            size_str = f"{size:.2f} MB"
                    return size_str

                message = self.encode_image(image)
                size = _get_size(message)
                if not message:
                    logger.error(f"{LP}publish: could not encode image, skipping...")
                    return

                self.client.publish(
                    f"{root_topic}/image/format",
                    self.config.image.format,
                    qos=self.config.qos,
                    retain=self.config.retain,
                )
                topic = f"{root_topic}/image/data"
                _msg = "N/A"
                if isinstance(message, bytes):
                    if self.config.image.format == "base64":
                        _msg = f"'<base64 encoded jpeg>'"
                        message = message.decode("utf-8")
                    else:
                        _msg = f"'<bytes encoded jpeg>'"

                logger.debug(
                    f"{LP}publish:IMG:{self.config.image.format}:: sending -> topic: '{topic}' data: {_msg} size: {size}"
                )
                self.client.publish(
                    topic, message, qos=self.config.qos, retain=self.config.image.retain
                )
        else:
            logger.debug(f"{LP}publish: image not enabled, skipping...")

    def close(self):
        if not self._connected:
            return
        try:
            if self.conn_time:
                self.conn_time = perf_counter() - self.conn_time
            logger.debug(
                f"{LP}close: {self.client_id} ->  disconnecting from mqtt broker: "
                f"'{self.config.broker if not self.sanitize else self.sanitize_str}:{self.config.port}'"
                f" [connection alive for: {self.conn_time} seconds]",
            )
            self.client.disconnect()
            self.client.loop_stop()
            self._connected = False
        except Exception as e:
            return logger.error(f"{LP}close:err_msg-> {e}")
