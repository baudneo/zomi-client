# A class that will represent a remote ML API
import asyncio
import atexit
import pickle
import time
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING, List
import logging

import aiohttp
from jose import jwt as jose_jwt

from ...Log import CLIENT_LOGGER_NAME
from ...Models.config import GlobalConfig
from ...main import get_global_config

if TYPE_CHECKING:
    from ...Models.config import ServerRoute

logger = logging.getLogger(CLIENT_LOGGER_NAME)
g = GlobalConfig()


class MLAPI:
    host: str
    port: int
    username: str
    password: str
    token: Optional[str] = None
    _session: Optional[aiohttp.ClientSession]
    lp: str = "mlapi:"
    tasks: List[asyncio.Task] = []

    def __init__(self, config: "ServerRoute"):
        global g

        g = get_global_config()
        self.config = config
        self.cached_token_path = (
            g.config.system.variable_data_path / "misc/.ml_token.pkl"
        )
        self.host = str(config.host)
        self.port = config.port
        self.name = config.name
        self.username = config.username
        self.password = config.password.get_secret_value()

        _ = self.cached_token
        atexit.register(self.clean_up)

    def clean_up(self):
        lp = f"{self.lp}clean_up:"
        try:
            for task in self.tasks:
                if not task.done():
                    task.cancel()
        except Exception as cleanup_exc:
            logger.exception("%s Error cleaning up: %s" % (lp, cleanup_exc))

    @staticmethod
    def decode_jwt(token: str) -> dict:
        decoded = {}
        try:
            decoded = jose_jwt.get_unverified_claims(token)
        except jose_jwt.JWTError as jwt_exc:
            logger.error(f"Error decoding JWT: {jwt_exc}")
        return decoded

    def write_cache(self, data: dict) -> bool:
        """
        Write the token to disk.

        :param data: The token data to write to disk
        :return: True if successful, False otherwise
        """
        try:
            with self.cached_token_path.open("wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving token to cache: {e}")
            return False
        return True

    def read_cache(self) -> dict:
        """Read the cached token from disk."""
        data = {}
        if self.cached_token_path.exists():
            try:
                with self.cached_token_path.open("rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading token from cache: {e}")

        return data

    def check_token(self, token) -> bool:
        _decoded = self.decode_jwt(token)
        _expire = _decoded.get("exp")
        if _expire:
            expire_at = datetime.fromtimestamp(float(_expire))
            logger.debug(f"{self.lp} token expires at: {expire_at}")

            if expire_at < datetime.now() - timedelta(minutes=5):
                logger.debug(f"{self.lp} token expired (or about to expire)")
                return False
            else:
                logger.debug(f"{self.lp} token should still be valid!")

        return True

    @property
    def cached_token(self) -> Optional[str]:
        tkn_pth = self.cached_token_path
        ml_token = ""
        if tkn_pth.exists():
            ml_token_data = self.read_cache()
            ml_token = ml_token_data.get("access_token")
        else:
            tkn_pth.touch(exist_ok=True, mode=0o640)

        return ml_token

    @cached_token.setter
    def cached_token(self, data: dict):
        try:
            pickle.dump(data, self.cached_token_path.open("wb"))
        except Exception as e:
            logger.error(f"{self.lp} Error saving token to cache: {e}")
        else:
            logger.debug(f"{self.lp} Token saved to cache: {self.cached_token_path}")

    @property
    def base_url(self) -> str:
        return f"{self.host.rstrip('/')}:{self.port}"

    async def login(self) -> Optional[str]:
        """
        Login to the ML API; the endpoint is /login.
        Provide a body request with username and password.
        """

        lp = f"{self.lp}login:"
        ml_token = None
        url = self.base_url + "/login"
        logger.debug(
            f"{lp} logging in @ "
            f"{url if not g.config.logging.sanitize.enabled else g.config.logging.sanitize.replacement_str}"
        )
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=10)
            ) as session:
                async with session.post(
                    url=url,
                    data={
                        "username": self.username,
                        "password": self.password,
                    },
                ) as r:
                    status = r.status
                    r.raise_for_status()
                    if status == 200:
                        # Response is always JSON
                        resp = await r.json()
                        if ml_token := resp.get("access_token"):
                            logger.info(f"{lp} login success!")
                            self.token = ml_token
                            self.cached_token = resp
                        else:
                            logger.error(f"{lp} login failed: {resp}")
                    else:
                        logger.error(
                            f"{lp} route '{self.name}' returned non-200 status: {status}\n{r}"
                        )
        except Exception as login_exc:
            logger.error(f"{lp} Error logging in to ML API: {login_exc}")

        return ml_token

    async def inference(self, images: str, hints: str) -> dict:
        """
        Send images to the ML API for inference.

        :param images: The images to send in a JSON formatted string
        :param hints: The hints to send in a JSON formatted string
        """
        if not self.token:
            await self.login()
        reply = {}
        url = self.base_url + "/detect"
        lp = f"{self.lp}inference:"
        # Multipart form data
        mp = aiohttp.FormData()
        mp.add_field("images", images, content_type="multipart/form-data")
        mp.add_field("model_hints", hints, content_type="multipart/form-data")
        logger.debug(
            f"{lp} sending data to 'Machine Learning API' ['{self.name}' @ "
            f"{url if not g.config.logging.sanitize.enabled else g.config.logging.sanitize.replacement_str}]"
        )
        _perf = time.time()
        r: aiohttp.ClientResponse
        mlapi_timeout = self.config.timeout
        if mlapi_timeout is None:
            mlapi_timeout = 90.0
        session: aiohttp.ClientSession = g.api.async_session
        try:
            async with session.post(
                url,
                data=mp,
                timeout=aiohttp.ClientTimeout(total=mlapi_timeout),
                headers={
                    "Authorization": f"Bearer {self.token}",
                },
            ) as r:
                r.raise_for_status()
                status = r.status
                if r.content_type == "application/json":
                    reply = await r.json()
                else:
                    logger.error(
                        f"{lp} route '{self.name}' returned a non-json response! \n{r}"
                    )
        # deal with unauthorized
        except aiohttp.ClientResponseError as e:
            logger.warning(f"{lp} API returned: {e}")
            if e.status == 401:
                logger.error(f"{lp} API returned 401 unauthorized!")

        except Exception as e:
            logger.error(f"{lp} Error sending image to API: {e}")

        logger.debug(
            f"perf:{lp} detection request to '{self.name}' completed in "
            f"{time.time() - _perf:.5f} seconds"
        )

        return reply
