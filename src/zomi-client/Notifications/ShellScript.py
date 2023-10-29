import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict

from ..Log import CLIENT_LOGGER_NAME
from ..Notifications import CoolDownBase
from ..main import get_global_config
from ..Models.config import GlobalConfig

logger = logging.getLogger(CLIENT_LOGGER_NAME)
g: Optional[GlobalConfig] = None
LP: str = "shell_script:"


class ShellScriptNotification(CoolDownBase):
    _data_dir_str: str = "push/shell_script"

    def __init__(self):
        global g

        g = get_global_config()
        self.config = g.config.notifications.shell_script
        self.data_dir = g.config.system.variable_data_path / self._data_dir_str
        super().__init__()

    def run(self, fmt_str: Optional[str] = None, results: Optional[Dict] = None):
        lp: str = f"{LP}:run:"
        script_path = Path(self.config.script)
        script_args = self.config.args
        if script_args is None:
            script_args = []
        _accepted_args = {
            "mid", "eid", "fmt_str", "event_url", "event_path", "results"
        }
        if self.config.enabled:
            for _arg in script_args:
                _arg = _arg.strip().casefold()
                if _arg == "mid":
                    script_args[script_args.index(_arg)] = str(g.mid)
                elif _arg == "eid":
                    script_args[script_args.index(_arg)] = str(g.eid)
                elif _arg == "fmt_str":
                    script_args[script_args.index(_arg)] = fmt_str
                elif _arg == "event_url":
                    script_args[script_args.index(_arg)] = (
                        f"{g.api.portal_base_url}/cgi-bin/nph-zms?mode=jpeg&replay=single&"
                        f"monitor={g.mid}&event={g.eid}"
                    )
                elif _arg == "event_path":
                    # replace using index
                    script_args[script_args.index(_arg)] = g.event_path.as_posix()
                elif _arg == "results":
                    import json

                    x = dict(results)
                    x.pop("frame_img")
                    script_args[script_args.index(_arg)] = json.dumps(x)
                else:
                    logger.debug(f"{lp} Removing unknown arg: {_arg}")
                    script_args.remove(_arg)
            if not script_path.is_file():
                logger.error(f"Script file '{script_path.as_posix()}' not found/is not a valid file")
            if self.config.I_AM_AWARE_OF_THE_DANGER_OF_RUNNING_SHELL_SCRIPTS != "YeS i aM awaRe!":
                logger.error("You MUST set I_AM_AWARE_OF_THE_DANGER_OF_RUNNING_SHELL_SCRIPTS to: YeS i aM awaRe!")
            else:
                if not script_path.is_absolute():
                    script_path = script_path.expanduser().resolve()
                cmd_array = [script_path.as_posix()]
                if script_args:
                    cmd_array.extend(script_args)
                x: Optional[subprocess.CompletedProcess] = None
                try:
                    x = subprocess.run(cmd_array, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"{lp} Shell script failed with exit code {e.returncode}")
                if x:
                    if x.stdout:
                        logger.debug(f"{lp} STDOUT->{x.stdout}")
                    if x.stderr:
                        logger.error(f"{lp} STDERR-> {x.stderr}")
        else:
            logger.debug(f"Shell script notification disabled, skipping")

    def send(self, *args, **kwargs):
        return self.run(*args, **kwargs)
