#!/usr/bin/env python3
import shutil
import venv
import platform
import subprocess
import tempfile
import time
from collections import namedtuple
import logging
import argparse
from typing import Optional, Tuple, Union, List, Dict, Pattern
from pathlib import Path
from threading import Thread
import re
import os
import sys

_simple_re: Pattern = re.compile(r"(?<!\\)\$([A-Za-z0-9_]+)")
_extended_re: Pattern = re.compile(r"(?<!\\)\$\{([A-Za-z0-9_]+)((:?-)([^}]+))?}")

# Change these if you want to install to a different default location
# You can also specify these locations via CLI
DEFAULT_DATA_DIR = "/opt/zomi/client"
DEFAULT_MODEL_DIR = f"{DEFAULT_DATA_DIR}/models"
DEFAULT_CONFIG_DIR = f"{DEFAULT_DATA_DIR}/conf"
DEFAULT_LOG_DIR = f"{DEFAULT_DATA_DIR}/logs"
DEFAULT_TMP_DIR = f"{tempfile.gettempdir()}/zomi/client"
DEFAULT_IMG_DIR = f"{DEFAULT_DATA_DIR}/images"

DEFAULT_SYSTEM_CREATE_PERMISSIONS = 0o755
# config files will have their permissions adjusted to this
DEFAULT_CONFIG_CREATE_PERMISSIONS = 0o755
# default ML models to install (SEE: available_models{})
REPO_BASE = Path(__file__).parent.parent
EXAMPLES_DIR = Path(__file__).parent
_ENV = {}
THREADS: Dict[str, Thread] = {}

# Logging
logger = logging.getLogger("install_client")
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
console = logging.StreamHandler(stream=sys.stdout)
console.setFormatter(log_formatter)
logger.addHandler(console)

# Misc.
__dependencies__ = "psutil", "requests", "tqdm", "distro"
__doc__ = """Install ZoMi Machine Learning Client - Custom"""

# Logic
tst_msg_wrap = "[testing!!]", "[will not actually execute]"


# parse .env file using pyenv
def parse_env_file(env_file: Path) -> None:
    """Parse .env file using python-dotenv.

    :param env_file: Path to .env file.
    :type env_file: Path
    :returns: Dict of parsed .env file.
    """
    try:
        import dotenv
    except ImportError:
        logger.warning(f"python-dotenv not installed, skipping {env_file}")
    else:
        global _ENV
        dotenv_vals = dotenv.dotenv_values(env_file)
        _ENV.update(dotenv_vals)


def test_msg(msg_: str, level: Optional[Union[str, int]] = None):
    """Print test message. Changes stack level to 2 to show caller of test_msg."""
    if testing:
        logger.warning(f"{tst_msg_wrap[0]} {msg_} {tst_msg_wrap[1]}", stacklevel=2)
    else:
        if level in ("debug", logging.DEBUG, None):
            logger.debug(msg_, stacklevel=2)
        elif level in ("info", logging.INFO):
            logger.info(msg_, stacklevel=2)
        elif level in ("warning", logging.WARNING):
            logger.warning(msg_, stacklevel=2)
        elif level in ("error", logging.ERROR):
            logger.error(msg_, stacklevel=2)
        elif level in ("critical", logging.CRITICAL):
            logger.critical(msg_, stacklevel=2)
        else:
            logger.info(msg_, stacklevel=2)


def get_distro() -> namedtuple:
    if hasattr(platform, "freedesktop_os_release"):
        release_data = platform.freedesktop_os_release()
    else:
        import distro

        release_data = {"ID": distro.id()}
    nmd_tpl = namedtuple("Distro", release_data.keys())
    return nmd_tpl(**release_data)


def check_imports():
    import importlib

    ret = True
    for imp_name in __dependencies__:
        try:
            importlib.import_module(imp_name)
        except ImportError:
            _msg = (
                f"Missing python module dependency: {imp_name}"
                f":: Please install the python package"
            )
            logger.error(_msg)
            print(_msg)
            ret = False
        else:
            logger.debug(f"Found python module dependency: {imp_name}")
    return ret


def get_web_user() -> Tuple[Optional[str], Optional[str]]:
    """Get the user that runs the web server using psutil library.

    :returns: The user that runs the web server.
    :rtype: str

    """
    import psutil
    import grp

    # get group name from gid

    www_daemon_names = ("httpd", "hiawatha", "apache", "nginx", "lighttpd", "apache2")
    hits = []
    proc_names = []
    for proc in psutil.process_iter():
        if any(x.startswith(proc.name()) for x in www_daemon_names):
            uname, ugroup = proc.username(), grp.getgrgid(proc.gids().real).gr_name
            logger.debug(f"Found web server process: {proc.name()} ({uname}:{ugroup})")
            hits.append((uname, ugroup))
            proc_names.append(proc.name())
    proc_names = list(set(proc_names))
    if len(hits) > 1:
        import pwd

        # sort by uid high to low
        hits = sorted(hits, key=lambda x: pwd.getpwnam(x[0]).pw_uid, reverse=True)
        logger.warning(
            f"Multiple web server processes found ({proc_names}) - The list is sorted to account "
            f"for root dropping privileges. Hopefully the first entry is the correct one -> {hits}"
        )

    if hits:
        return hits[0]
    return None, None


def parse_cli():
    global args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlapi-user", help="API user", type=str, default=None, dest="api_user")
    parser.add_argument("--mlapi-pass", help="API password", type=str, default=None, dest="api_pass")
    parser.add_argument(
        "--no-cache-dir",
        action="store_true",
        dest="no_cache",
        help="Do not use cache directory for pip install",
    )
    parser.add_argument(
        "--editable",
        action="store_true",
        dest="pip_install_editable",
        help="Use the --editable flag when installing via pip (a git pull will update "
        "the installed package) BE AWARE: you must keep the git source directory intact for this to work",
    )
    parser.add_argument(
        "--zm-portal",
        dest="zm_portal",
        default=None,
        type=str,
        help="ZoneMinder portal URL [Optional]",
    )
    parser.add_argument(
        "--zm-api",
        dest="zm_api",
        default=None,
        type=str,
        help="ZoneMinder API URL, If this is omited and --zm-portal "
        "is specified, the API URL will be derived from the portal URL (--zm-portal + '/api')",
    )
    parser.add_argument(
        "--zm-user",
        dest="zm_user",
        default=None,
        type=str,
        help="ZoneMinder API user [Optional]",
    )
    parser.add_argument(
        "--zm-pass",
        dest="zm_pass",
        default=None,
        type=str,
        help="ZoneMinder API password [Optional]",
    )
    parser.add_argument(
        "--route-name",
        dest="route_name",
        default=None,
        type=str,
        help="MLAPI route name [Optional]",
    )
    # address and port as well as the route name
    parser.add_argument(
        "--route-host",
        dest="route_host",
        default=None,
        type=str,
        help="MLAPI host address [Optional]",
    )
    parser.add_argument(
        "--route-port",
        dest="route_port",
        default=None,
        type=str,
        help="MLAPI host port [Optional]",
    )

    parser.add_argument(
        "--config-only",
        dest="config_only",
        action="store_true",
        help="Install config files only, used in conjunction with --secret-only",
    )

    parser.add_argument(
        "--secrets-only",
        dest="secrets_only",
        action="store_true",
        help="Install secrets file only, used in conjunction with --config-only",
    )

    parser.add_argument(
        "--interactive",
        "-I",
        action="store_true",
        dest="interactive",
        help="Run in interactive mode",
    )

    parser.add_argument(
        "--install-log",
        type=str,
        default=f"./zomi-client_install.log",
        help="File to write installation logs to",
    )
    parser.add_argument(
        "--env-file", type=Path, help="Path to .env file (requires python-dotenv)"
    )
    parser.add_argument(
        "--dir-config",
        help=f"Directory where config files are held Default: {DEFAULT_CONFIG_DIR}",
        default=DEFAULT_CONFIG_DIR,
        type=Path,
        dest="config_dir",
    )
    parser.add_argument(
        "--dir-data",
        help=f"Directory where variable data is held Default: {DEFAULT_DATA_DIR}",
        dest="data_dir",
        default=DEFAULT_DATA_DIR,
        type=Path,
    )
    parser.add_argument(
        "--dir-model",
        type=str,
        help="ML model base directory",
        default=DEFAULT_MODEL_DIR,
        dest="model_dir",
    )
    parser.add_argument(
        "--dir-image",
        "--dir-images",
        type=str,
        help="Directory where images are stored",
        default=DEFAULT_IMG_DIR,
        dest="image_dir",
    )
    parser.add_argument(
        "--dir-temp",
        "--dir-tmp",
        type=Path,
        help="Temp files directory",
        default=DEFAULT_TMP_DIR,
        dest="tmp_dir",
    )
    parser.add_argument(
        "--dir-log",
        help=f"Directory where logs will be stored Default: {DEFAULT_LOG_DIR}",
        default=DEFAULT_LOG_DIR,
        dest="log_dir",
        type=Path,
    )
    parser.add_argument(
        "--user",
        "-U",
        help="User to install as [leave empty to auto-detect what user runs "
        "the web server] (Change if installing server on a remote host)",
        type=str,
        dest="ml_user",
        default="",
    )
    parser.add_argument(
        "--group",
        "-G",
        help="Group member to install as [leave empty to auto-detect what "
        "group member runs the web server] (Change if installing server on a remote host)",
        type=str,
        dest="ml_group",
        default="",
    )

    parser.add_argument(
        "--debug", "-D", help="Enable debug logging", action="store_true", dest="debug"
    )
    parser.add_argument(
        "--dry-run",
        "--test",
        "-T",
        action="store_true",
        dest="test",
        help="Run in test mode, no actions are actually executed.",
    )

    parser.add_argument(
        "--system-create-permissions",
        help=f"ZM ML system octal [0o] file permissions [Default: {oct(DEFAULT_SYSTEM_CREATE_PERMISSIONS)}]",
        type=lambda x: int(x, 8),
        default=DEFAULT_SYSTEM_CREATE_PERMISSIONS,
    )
    parser.add_argument(
        "--config-create-permissions",
        help=f"Config files (server.yml, client.yml, secrets.yml) octal [0o] file permissions "
        f"[Default: {oct(DEFAULT_CONFIG_CREATE_PERMISSIONS)}]",
        type=lambda x: int(x, 8),
        default=DEFAULT_CONFIG_CREATE_PERMISSIONS,
    )

    return parser.parse_args()


def chown(path: Path, user: Union[str, int], group: Union[str, int]):
    import pwd
    import grp

    if isinstance(user, str):
        user = pwd.getpwnam(user).pw_uid
    if isinstance(group, str):
        group = grp.getgrnam(group).gr_gid
    test_msg(f"chown {user}:{group} {path}")
    if not testing:
        os.chown(path, user, group)


def chmod(path: Path, mode: int):
    test_msg(f"chmod OCTAL: {mode:o} RAW: {mode} => {path}")
    if not testing:
        os.chmod(path, mode)


def chown_mod(path: Path, user: Union[str, int], group: Union[str, int], mode: int):
    chown(path, user, group)
    chmod(path, mode)


def create_dir(path: Path, user: Union[str, int], group: Union[str, int], mode: int):
    message = (
        f"Created directory: {path} with user:group:permission "
        f"[{user}:{group}:{mode:o}]"
    )
    test_msg(message)
    if not testing:
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        chown_mod(path, user, group, mode)


def show_config(cli_args: argparse.Namespace):
    logger.info(f"Configuration:")
    for key, value in vars(cli_args).items():
        if key.endswith("_permissions"):
            value = oct(value)
        elif key == "models":
            value = ", ".join(value)
        elif isinstance(value, Path):
            value = value.expanduser().resolve().as_posix()
        logger.info(f"    {key}: {value.__repr__()}")
    if args.interactive:
        input(
            "Press Enter to continue if all looks fine, otherwise use 'Ctrl+C' to exit and edit CLI options..."
        )
    else:
        _msg_ = f"This is a non-interactive{' TEST' if testing else ''} session, continuing with installation... "
        if testing:
            logger.info(_msg_)
        else:
            logger.info(f"{_msg_} in 5 seconds, press 'Ctrl+C' to exit")
            time.sleep(5)


def do_web_user():
    global ml_user, ml_group
    _group = None
    if not ml_user or not ml_group:
        if not ml_user:
            if interactive:
                ml_user = input("Web server username [Leave blank to try auto-find]: ")
                if not ml_user:
                    ml_user, _group = get_web_user()
                    logger.info(f"Web server user auto-find: {ml_user}")
            else:
                ml_user, _group = get_web_user()
                logger.info(f"Web server user auto-find: {ml_user}")

        if not ml_group:
            if interactive:
                ml_group = input("Web server group [Leave blank to try auto-find]: ")
                if not ml_group:
                    if _group:
                        ml_group = _group
                    else:
                        _, ml_group = get_web_user()
                    logger.info(f"Web server group auto-find: {ml_group}")

            else:
                if _group:
                    ml_group = _group
                else:
                    _, ml_group = get_web_user()
                logger.info(f"Web server group auto-find: {ml_group}")

    if not ml_user or not ml_group:
        _missing = ""
        _u = "user [--user]"
        _g = "group [--group]"
        if not ml_user and ml_group:
            _missing = _u
        elif ml_user and not ml_group:
            _missing = _g
        else:
            _missing = f"{_u} and {_g}"

        _mess = f"Unable to determine web server {_missing}, EXITING..."
        if not testing:
            logger.error(_mess)
            sys.exit(1)
        else:
            test_msg(_mess)


def install_dirs(
    dest_dir: Path,
    default_dir: str,
    dir_type: str,
    sub_dirs: Optional[List[str]] = None,
    perms: int = 0o755,
):
    """Create directories with sub dirs"""
    if sub_dirs is None:
        sub_dirs = []
    if not dest_dir:
        if interactive:
            dest_dir = input(
                f"Set {dir_type} directory [Leave blank to use default {default_dir}]: "
            )

    if not dest_dir:
        if default_dir:
            logger.info(f"Using default {dir_type} directory: {default_dir}")
            dest_dir = default_dir
        else:
            raise ValueError(f"Default {dir_type} directory not set!")

    dest_dir = Path(dest_dir)
    if dir_type.casefold() == "data":
        global data_dir
        data_dir = dest_dir
    elif dir_type.casefold() == "log":
        global log_dir
        log_dir = dest_dir
    elif dir_type.casefold() == "config":
        global cfg_dir
        cfg_dir = dest_dir

    if not dest_dir.exists():
        logger.warning(f"{dir_type} directory {dest_dir} does not exist!")
        if interactive:
            x = input(f"Create {dir_type} directory [Y/n]?... 'n' will exit! ")
            if x.strip().casefold() == "n":
                logger.error(f"{dir_type} directory does not exist, exiting...")
                sys.exit(1)
            create_dir(dest_dir, ml_user, ml_group, perms)
        else:
            test_msg(f"Creating {dir_type} directory...")
            create_dir(dest_dir, ml_user, ml_group, perms)
    else:
        logger.warning(f"{dir_type} directory {dest_dir} already exists!")
    # create sub-folders
    if sub_dirs:
        test_msg(
            f"Creating {dir_type} sub-folders: {', '.join(sub_dirs).lstrip(',')}"
        )
        for _sub in sub_dirs:
            _path = dest_dir / _sub
            create_dir(_path, ml_user, ml_group, perms)


def download_file(url: str, dest: Path, user: str, group: str, mode: int):
    _mess_ = (
        f"Downloading {url}..."
        if not testing
        else f"TESTING if file exists at :: {url}..."
    )
    logger.info(_mess_)

    import requests
    from tqdm.auto import tqdm
    import shutil
    import functools

    try:
        r = requests.get(url, stream=True, allow_redirects=True, timeout=5)
        if r:
            file_size = int(r.headers.get("Content-Length", 0))
            dest = dest.expanduser().resolve()
            dest.parent.mkdir(parents=True, exist_ok=True)

            desc = (
                "(Unknown total file size!)"
                if file_size == 0
                else f"Downloading {url} ..."
            )
            r.raw.read = functools.partial(
                r.raw.read, decode_content=True
            )  # Decompress if needed
            with tqdm.wrapattr(
                r.raw, "read", total=file_size, desc=desc, colour="green"
            ) as r_raw:
                do_chown = True
                if testing:
                    do_chown = False
                    logger.info(
                        f"TESTING: File exists at the url for {url.split('/')[-1]}"
                    )
                    # to keep the progress bar, pipe output to /dev/null
                    dest = Path("/dev/null")
                try:
                    with dest.open("wb") as f:
                        shutil.copyfileobj(r_raw, f)
                except Exception as e:
                    logger.error(
                        f"Failed to open or copy data to destination file ({dest}) => {e}"
                    )
                    raise e
                else:
                    logger.info(f"Successfully downloaded {url} to {dest}")
                    if do_chown:
                        chown_mod(dest, user, group, mode)
        else:
            logger.error(f"NO RESPONSE FROM {url} :: Failed to download {url}!")
    except requests.exceptions.ConnectionError:
        logger.error(f"REQUESTS CONNECTION ERROR :: Failed to download {url}!")
        return
    except Exception as e:
        logger.error(f"Failed to download {url}! EXCEPTION :: {e}")


def copy_file(src: Path, dest: Path, user: str, group: str, mode: int):
    """Copy a file from src to dest, chown and chmod it using user:group and mode."""
    __msg = f"Copying {src} to {dest}..."
    if not testing:
        # Check if the file exists, if so, show a log warning and remove, then copy
        import shutil

        if dest.exists():
            logger.warning(f"File {dest} already exists, removing...")
            dest.unlink()

        logger.info(__msg)
        shutil.copy(src, dest)
        chown_mod(dest, user, group, mode)
    else:
        test_msg(__msg)


def get_pkg_manager():
    distro = get_distro()
    binary, prefix = "apt", ["install", "-y"]
    if distro.ID in ("debian", "ubuntu", "raspbian"):
        pass
    elif distro.ID in ("centos", "fedora", "rhel"):
        binary = "yum"
    elif distro.ID == "fedora":
        binary = "dnf"
    elif distro.ID in ("arch", "manjaro"):
        binary = "pacman"
        prefix = ["-S", "--noconfirm"]
    elif distro.ID == "gentoo":
        binary = "emerge"
        prefix = ["-av"]
    elif distro.ID == "alpine":
        binary = "apk"
        prefix = ["add", "-q"]
    elif distro.ID == "suse":
        binary = "zypper"
    elif distro.ID == "void":
        binary = "xbps-install"
        prefix = ["-y"]
    elif distro.ID == "nixos":
        binary = "nix-env"
        prefix = ["-i"]
    elif distro.ID == "freebsd":
        binary = "pkg"
    elif distro.ID == "openbsd":
        binary = "pkg_add"
        prefix = []
    elif distro.ID == "netbsd":
        binary = "pkgin"
    elif distro.ID == "solus":
        binary = "eopkg"
    elif distro.ID == "windows":
        binary = "choco"
    elif distro.ID == "macos":
        binary = "brew"

    return binary, prefix


def install_host_dependencies(_type: str):
    _type = _type.strip().casefold()
    if _type == "secrets":
        return
    if _type not in ["server", "client"]:
        logger.error(f"Invalid type '{_type}'")
    else:
        inst_binary, inst_prefix = get_pkg_manager()
        dependencies = {
            "apt": {
                "client": {
                    "binary_names": ["gifsicle", "geos-config"],
                    "binary_flags": ["--version", "--version"],
                    "pkg_names": ["gifsicle", "libgeos-dev"],
                },
                "server": {
                    "binary_names": [],
                    "binary_flags": [],
                    "pkg_names": [],
                },
            },
            "yum": {
                "client": {
                    "binary_names": ["gifsicle", "geos-config"],
                    "binary_flags": ["--version", "--version"],
                    "pkg_names": ["gifsicle", "geos-devel"],
                },
                "server": {
                    "binary_names": [],
                    "binary_flags": [],
                    "pkg_names": [],
                },
            },
            "pacman": {
                "client": {
                    "binary_names": ["gifsicle", "geos-config"],
                    "binary_flags": ["--version", "--version"],
                    "pkg_names": ["gifsicle", "geos"],
                },
                "server": {
                    "binary_names": [],
                    "binary_flags": [],
                    "pkg_names": [],
                },
            },
            "zypper": {
                "client": {
                    "binary_names": ["gifsicle", "geos-config"],
                    "binary_flags": ["--version", "--version"],
                    "pkg_names": ["gifsicle", "geos"],
                },
                "server": {
                    "binary_names": [],
                    "binary_flags": [],
                    "pkg_names": [],
                },
            },
            "dnf": {
                "client": {
                    "binary_names": ["gifsicle", "geos-config"],
                    "binary_flags": ["--version", "--version"],
                    "pkg_names": ["gifsicle", "geos-devel"],
                },
                "server": {
                    "binary_names": [],
                    "binary_flags": [],
                    "pkg_names": [],
                },
            },
            "apk": {
                "client": {
                    "binary_names": ["gifsicle", "geos-config"],
                    "binary_flags": ["--version", "--version"],
                    "pkg_names": ["gifsicle", "geos-devel"],
                },
                "server": {
                    "binary_names": [],
                    "binary_flags": [],
                    "pkg_names": [],
                },
            },
        }
        deps = []
        deps_cmd = []

        if _type == "server":
            _msg = "Installing server HOST dependencies..."
            test_msg(_msg)
            full_deps = zip(
                dependencies[inst_binary]["server"]["pkg_names"],
                dependencies[inst_binary]["server"]["binary_names"],
                dependencies[inst_binary]["server"]["binary_flags"],
            )
            for _dep, _bin, _flag in full_deps:
                deps_cmd.append([_bin, _flag])
                deps.append(_dep)
        elif _type == "client":
            _msg = "Testing for client HOST dependencies..."
            test_msg(_msg)
            full_deps = zip(
                dependencies[inst_binary]["client"]["pkg_names"],
                dependencies[inst_binary]["client"]["binary_names"],
                dependencies[inst_binary]["client"]["binary_flags"],
            )
            for _dep, _bin, _flag in full_deps:
                deps_cmd.append([_bin, _flag])
                deps.append(_dep)
        else:
            logger.error(f"Invalid type '{_type}'")
            return

        def test_cmd(cmd_array: List[str], dep_name: str):
            logger.debug(
                f"Testing if dependency {_dep_name} is installed by running: {' '.join(cmd_array)}"
            )

            try:
                x = subprocess.run(cmd_array, check=True, capture_output=True)
            except subprocess.CalledProcessError as proc_err:
                logger.error(f"Error while running {cmd_array} -> {proc_err}")
            except FileNotFoundError:
                logger.error(
                    f"Failed to locate {cmd_array[0]} please install HOST package: {dep_name}"
                )
                raise

            except Exception as exc_:
                logger.error(f"Exception type: {type(exc_)} --- {exc_}")
                raise exc_
            else:
                logger.info(f"{cmd_array[0]} is installed")
                logger.debug(f"{cmd_array} output: {x.stdout.decode('utf-8')}")

        if deps:
            full_deps = zip(deps_cmd, deps)
            for _cmd_array, _dep_name in full_deps:
                try:
                    test_cmd(_cmd_array, _dep_name)
                except FileNotFoundError:
                    msg = f"package '{_dep_name}' is not installed, please install it manually!"
                    if os.geteuid() != 0 and not testing:
                        logger.warning(
                            f"You must be root to install host dependencies! {msg}"
                        )
                    else:
                        if testing:
                            logger.warning(
                                f"Running as non-root user but this is test mode! continuing to test... "
                            )
                        install_cmd = [inst_binary, inst_prefix, _dep_name]
                        msg = f"Running HOST package manager installation command: {install_cmd}"
                        if not interactive:
                            if not testing:
                                logger.info(msg)
                                try:
                                    subprocess.run(
                                        install_cmd,
                                        check=True,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                    )
                                except subprocess.CalledProcessError as e:
                                    logger.error(
                                        f"Error installing host dependencies: {e.stdout.decode('utf-8')}"
                                    )
                                    sys.exit(1)
                                else:
                                    logger.info(
                                        "Host dependencies installed successfully"
                                    )
                            else:
                                test_msg(msg)
                        else:
                            if not testing:
                                _install = False
                                _input = input(
                                    f"Host dependencies are not installed, would you like to install "
                                    f"{' '.join(deps)}? [Y/n]"
                                )
                                if _input:
                                    _input = _input.strip().casefold()
                                    if _input == "n":
                                        logger.warning(
                                            "Host dependencies not installed, please install them manually!"
                                        )
                                    else:
                                        _install = True
                                else:
                                    _install = True
                                if _install:
                                    try:
                                        subprocess.run(
                                            install_cmd,
                                            check=True,
                                        )
                                    except subprocess.CalledProcessError as e:
                                        logger.error(
                                            f"Error installing host dependencies: {e.stdout.decode('utf-8')}"
                                        )
                                        sys.exit(1)
                                    else:
                                        logger.info(
                                            "Host dependencies installed successfully"
                                        )


def main():
    install_dirs(
        data_dir,
        DEFAULT_DATA_DIR,
        "Data",
        sub_dirs=[
            "models",
            "push",
            "scripts",
            "images",
            "bin",
            "misc",
        ],
    )
    install_dirs(
        Path(f"{DEFAULT_DATA_DIR}/face_data"),
        f"{DEFAULT_DATA_DIR}/face_data",
        "Face Data",
        sub_dirs=["known", "unknown"],
        perms=0o777,
    )
    install_dirs(
        cfg_dir, DEFAULT_CONFIG_DIR, "Config", sub_dirs=[], perms=cfg_create_mode
    )
    install_dirs(log_dir, DEFAULT_LOG_DIR, "Log", sub_dirs=[], perms=0o777)

    check_backup("secrets")
    do_install()


def check_backup(_inst_type: str) -> None:
    files: List[Optional[Path]] = [x for x in cfg_dir.rglob(f"{_inst_type}.*")]
    if files:
        files = sorted(files, reverse=True)
        _target: Optional[Path] = None
        backup_num = 1
        high = 0
        for _file in files:
            if _file.name.startswith(f"{_inst_type}"):
                if _file.suffix == ".bak":
                    i = int(_file.stem.split(".")[-1])
                    logger.debug(f"Found a {_inst_type} backup numbered: {i}")

                    if i > high:
                        high = backup_num = i + 1
                        logger.debug(
                            f"This backup has a higher number, using it as index: {backup_num = }"
                        )

                elif _file.suffix in (".yml", ".yaml"):
                    logger.debug(f"found existing {_inst_type} config file: {_file}")
                    _target = _file
        if _target:
            file_backup = cfg_dir / f"{_inst_type}.{backup_num}.bak"
            logger.debug(
                f"Backing up existing {_inst_type} config file: {_target} to {file_backup}"
            )
            # Backup existing
            copy_file(_target, file_backup, ml_user, ml_group, cfg_create_mode)
            # install new
            copy_file(
                REPO_BASE / f"configs/example_{_inst_type}.yml",
                cfg_dir / f"{_inst_type}.yml",
                ml_user,
                ml_group,
                cfg_create_mode,
            )
    else:
        logger.debug(f"No existing {_inst_type} config files found!")
        if cfg_dir.is_dir() or testing:
            test_msg(
                f"Creating {_inst_type} config file from example template...", "info"
            )
            copy_file(
                REPO_BASE / f"configs/example_{_inst_type}.yml",
                cfg_dir / f"{_inst_type}.yml",
                ml_user,
                ml_group,
                cfg_create_mode,
            )
        else:
            logger.warning(
                f"Config directory '{cfg_dir}' does not exist, skipping creation of {_inst_type} config file..."
            )


def do_install():
    global _ENV

    _inst_type: str = "client"
    check_backup("client")

    _cmd_array: List[str] = [
        "-m",
        "pip",
        "install",
    ]

    if args.no_cache:
        logger.info("Disabling pip cache...")
        _cmd_array.append("--no-cache-dir")
    if editable:
        logger.info(
            "Installing in pip --editable mode, DO NOT remove the source git"
            " directory after install! git pull will update the installed package"
        )
        _cmd_array.append("--editable")

    install_host_dependencies(_inst_type)

    copy_file(
        EXAMPLES_DIR / "EventStartCommand.sh",
        data_dir / "bin/EventStartCommand.sh",
        ml_user,
        ml_group,
        cfg_create_mode,
    )

    copy_file(
        EXAMPLES_DIR / "eventproc.py",
        data_dir / "bin/eventproc.py",
        ml_user,
        ml_group,
        cfg_create_mode,
    )

    # create a symbolic link to both files in /usr/local/bin
    # so that they can be called from anywhere
    test_msg(
        f"Creating symlinks for event start/stop commands: /usr/local/bin will contain zomi-ESC "
        f"(shell helper) zomi-eventproc (python script)",
        "info",
    )
    if not testing:
        # Make sure it does not exist first, it will error if file exists
        _dest_esc = Path("/usr/local/bin/zomi-ESC")
        _dest_ep = Path("/usr/local/bin/zomi-eventproc")
        # if it exists log a message that it is being unlinked then symlinked again with specified user

        if _dest_esc.exists():
            logger.warning(
                f"{_dest_esc} already exists, unlinking and sym-linking again..."
            )
            _dest_esc.unlink()
        _dest_esc.symlink_to(f"{data_dir}/bin/EventStartCommand.sh")
        if _dest_ep.exists():
            logger.warning(
                f"{_dest_ep} already exists, unlinking and sym-linking again..."
            )
            _dest_ep.unlink()
        _dest_ep.symlink_to(f"{data_dir}/bin/eventproc.py")

    _src: str = (
        f"{REPO_BASE.expanduser().resolve().as_posix()}"
    )

    create_("secrets", cfg_dir / "secrets.yml")
    create_(_inst_type, cfg_dir / f"{_inst_type}.yml")

    if testing and not editable:
        _cmd_array.append("--dry-run")
    _cmd_array.append(_src)

    _venv = ZoMiEnvBuilder(
        with_pip=True, cmd=_cmd_array, upgrade_deps=True, prompt="ZoMi_Client"
    )
    try:
        _venv.create(venv_dir)
    except FileNotFoundError as e:
        logger.warning(f"Issue while creating VENV: {e}")

    _f: Path = data_dir / "bin/eventproc.py"
    test_msg(
        f"Modifying {_f.as_posix()} to use VENV {_venv.context.env_exec_cmd} shebang"
    )
    if not testing:
        content: Optional[str] = None
        if not _f.is_absolute():
            _f = _f.expanduser().resolve()
        content = _f.read_text()
        with _f.open("w+") as f:
            f.write(f"#!{_venv.context.env_exec_cmd}\n{content}")
        del content


class Envsubst:
    strict: bool
    env: Optional[Dict]

    def __init__(self):
        """
        Substitute environment variables in a string. Instantiate and pass a string to the sub method.
        Modified to allow custom environment mappings, and to allow for strict mode (only use the custom environment mapping).

        Possibly add dotenv support?.
        """
        self.strict = False
        self.env = None

    def sub(self, search_string: str, env: Optional[Dict] = None, strict: bool = False):
        """
        Substitute environment variables in the given string, allows for passing a custom environment mapping.
        The default behavior is to check the custom environment mapping , the system environment and finally the
        specified default ("somestring" in the examples). If strict is True, the system environment will not be
        checked after the custom environment mapping but, the default will still be used (if needed).

        The following forms are supported:

        Simple variables - will use an empty string if the variable is unset and strict is true
          $FOO

        Bracketed expressions
          ${FOO}
            identical to $FOO
          ${FOO:-somestring}
            uses "somestring" if $FOO is unset, or is set and empty
          ${FOO-somestring}
            uses "somestring" only if $FOO is unset
        """
        self.strict = strict
        if strict:
            logger.info(f"envsubst:: strict mode: {self.strict}")
        if env:
            self.env = env
        # handle simple un-bracketed env vars like $FOO
        a: str = _simple_re.sub(self._repl_simple_env_var, search_string)
        logger.debug(f"envsubst DBG>>> after simple sub {type(a) = }")
        # handle bracketed env vars with optional default specification
        b: str = _extended_re.sub(self._repl_extended_env_var, a)
        return b

    def _resolve_var(self, var_name, default=None):
        if not self.strict and default is None:
            # Instead of returning an empty string in strict mode,
            # return the variable name formatted for zomi substitution
            default = f"${{{var_name}}}"

        if self.env:
            if not self.strict:
                return self.env.get(var_name, os.environ.get(var_name, default))
            else:
                return self.env.get(var_name, default)

        return os.environ.get(var_name, default)

    def _repl_simple_env_var(self, m: re.Match):
        var_name = m.group(1)
        return self._resolve_var(var_name, "" if self.strict else None)

    def _repl_extended_env_var(self, m: re.Match):
        if m:
            # ('ML_INSTALL_DATA_DIR', None, None, None)
            var_name = m.group(1)
            default_spec = m.group(2)
            if default_spec:
                default = m.group(4)
                default = _simple_re.sub(self._repl_simple_env_var, default)
                if m.group(3) == ":-":
                    # use default if var is unset or empty
                    env_var = self._resolve_var(var_name)
                    if env_var:
                        return env_var
                    else:
                        return default
                elif m.group(3) == "-":
                    # use default if var is unset
                    return self._resolve_var(var_name, default)
                else:
                    raise RuntimeError("unexpected string matched regex")
            else:
                return self._resolve_var(var_name, "" if self.strict else None)


def envsubst(string: str, env: Optional[Dict] = None, strict: bool = False):
    """Wraps Envsubst class for easier use.

    Args:
        string (str): String to substitute
        env (Optional[Dict]): Custom environment mapping. Defaults to None.
        strict (bool, optional): If True, only use the custom environment mapping instead of the current users ENV and the custom env mapping. Defaults to False.
    """
    _ = Envsubst()
    return _.sub(string, env=env, strict=strict)


def create_(_type: str, dest: Path):
    src = REPO_BASE / f"configs/example_{_type}.yml"
    envsubst_out = envsubst(src.read_text(), _ENV)
    test_msg(f"Writing {_type.upper()} file to {dest.as_posix()}")
    if not testing:
        dest.expanduser().resolve().write_text(envsubst_out)


def in_venv():
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


class ZoMiEnvBuilder(venv.EnvBuilder):
    """
    Venv builder for ZoMi Client.

    :param cmd: The pip command as an array.
    """

    install_cmd: List[str]
    context = None

    def __init__(self, *args, **kwargs):
        self.install_cmd = kwargs.pop("cmd", None)
        assert isinstance(
            self.install_cmd, list
        ), f"cmd must be a list not {type(self.install_cmd)}"
        assert self.install_cmd, f"cmd must not be empty!"
        super().__init__(*args, **kwargs)

    def post_setup(self, context):
        """
        Set up any packages which need to be pre-installed into the
        environment being created.
        :param context: The information for the environment creation request
                        being processed.
        """
        self.context = context
        old_env = os.environ.get("VIRTUAL_ENV", None)
        os.environ["VIRTUAL_ENV"] = context.env_dir
        self.install_zomi_client(context)
        if old_env:
            os.environ["VIRTUAL_ENV"] = old_env
        else:
            os.environ.pop("VIRTUAL_ENV")

    def install_zomi_client(self, context):
        """
        Install zomi in the environment.
        :param context: The information for the environment creation request
                        being processed.
        """

        self.install_cmd.insert(0, context.env_exec_cmd)
        logger.debug(
            f"venv builder:DBG>>> About to run install command: '{self.install_cmd}'"
        )
        remove_after = False
        if testing and editable:
            logger.info(f"***************************\n\n--editable and --dry-run/-T/--testing cannot be used "
                        f"together, the local --editable copy will be installed and then removed to accomplish "
                        f"the same thing. Resuming in 5 seconds...\n\n")
            time.sleep(5)

            remove_after = True
        ran: Optional[subprocess.Popen] = None
        try:
            ran = subprocess.Popen(self.install_cmd, stdout=subprocess.PIPE)

        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing pip dependencies!")
            logger.error(e)
            if e.stderr:
                logger.error(e.stderr)
            if e.stdout:
                logger.info(e.stdout)
            raise e
        else:
            if ran is not None:
                msg = ""
                for c in iter(lambda: ran.stdout.read(1), b""):
                    sys.stdout.buffer.write(c)
                    try:
                        c = c.decode("utf-8")
                    except UnicodeDecodeError:
                        pass
                    else:
                        # add to the msg until we get a newline
                        if c == "\n":
                            logger.info(msg)
                            msg = ""
                        else:
                            msg += c
        finally:
            if (remove_after and ran is not None) and Path(context.env_dir).exists():
                logger.info(f"Removing local --editable copy...")
                try:
                    shutil.rmtree(context.env_dir, ignore_errors=True)
                except Exception as e:
                    logger.error(
                        f"Failed to remove local --editable copy at {context.env_dir} -> {e}"
                    )
                else:
                    logger.info(f"Local --editable copy removed successfully")


def check_python_version(maj: int, min: int):
    if sys.version_info.major < maj:
        logger.error("Python 3+ is required to run this install script!")
        sys.exit(1)
    elif sys.version_info.major == maj:
        if sys.version_info.minor < min:
            logger.error(f"Python {maj}.{min}+ is required!")
            sys.exit(1)


if __name__ == "__main__":
    # check python is 3.8+ only
    check_python_version(3, 8)

    # parse args first
    args = parse_cli()

    logger.info(f"Starting zomi-client install script...")
    if not check_imports():
        logger.critical(f"Missing python dependencies, exiting...")
        sys.exit(1)
    else:
        logger.info("All python dependencies that this install script requires found.")

    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            from tqdm.auto import tqdm

            try:
                msg = self.format(record)
                # msg = f"TQDM::>>> {msg}"
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    tqdm_handler: TqdmLoggingHandler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(log_formatter)
    logger.removeHandler(console)
    logger.addHandler(tqdm_handler)
    testing: bool = args.test
    editable: bool = args.pip_install_editable
    debug: bool = args.debug
    install_log = args.install_log
    if in_venv():
        logger.info(
            "Detected to be running in a virtual environment, "
            "be aware this install script creates "
            "a venv for zomi to run in!"
        )
    file_handler = logging.FileHandler(install_log, mode="w")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    if debug:
        logger.setLevel(logging.DEBUG)
        for _handler in logger.handlers:
            _handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled!")
    if testing:
        logger.warning(">>>>>>>>>>>>> Running in test/dry-run mode! <<<<<<<<<<<<<<<<")

    system_create_mode: int = args.system_create_permissions
    cfg_create_mode: int = args.config_create_permissions
    interactive: bool = args.interactive
    data_dir: Path = args.data_dir
    cfg_dir: Path = args.config_dir
    log_dir: Path = args.log_dir
    model_dir: Path = args.model_dir
    image_dir: Path = args.image_dir
    tmp_dir: Path = args.tmp_dir
    venv_dir: Path = data_dir / "venv"
    zm_user = args.zm_user
    zm_pass = args.zm_pass
    mlapi_user = args.api_user
    mlapi_pass = args.api_pass
    zm_portal = args.zm_portal
    zm_api = args.zm_api
    if zm_portal and not zm_api:
        args.zm_api = zm_api = f"{zm_portal}/api"
        logger.info(
            f"--zm-portal specified, but not --zm-api. Appending '/api' -> {zm_api}"
        )
    route_name = args.route_name
    route_host = args.route_host
    route_port = args.route_port

    args.data_dir = data_dir = data_dir.expanduser().resolve()
    args.config_dir = cfg_dir = cfg_dir.expanduser().resolve()
    args.log_dir = log_dir = log_dir.expanduser().resolve()
    args.tmp_dir = tmp_dir = tmp_dir.expanduser().resolve()
    ml_user, ml_group = args.ml_user or "imoz", args.ml_group or "zomi"
    do_web_user()
    if not ml_user:
        logger.error(
            "user not specified/cant be computed (try to specify user and group), exiting..."
        )
        sys.exit(1)
    args.ml_user = ml_user
    args.ml_group = ml_group

    show_config(args)

    _ENV = {
        "ML_INSTALL_DATA_DIR": data_dir.as_posix(),
        "ML_INSTALL_VENV_DIR": venv_dir.as_posix(),
        "ML_INSTALL_CFG_DIR": cfg_dir.as_posix(),
        "ML_INSTALL_LOGGING_DIR": log_dir.as_posix(),
        "ML_INSTALL_LOGGING_LEVEL": "debug",
        "ML_INSTALL_LOGGING_CONSOLE_ENABLED": "yes",
        "ML_INSTALL_LOGGING_FILE_ENABLED": "no",
        "ML_INSTALL_MODEL_DIR": model_dir if model_dir else (data_dir / "models").as_posix(),
        "ML_INSTALL_LOGGING_SYSLOG_ENABLED": "no",
        "ML_INSTALL_LOGGING_SYSLOG_ADDRESS": "/dev/log",
        "ML_INSTALL_TMP_DIR": tmp_dir.as_posix() if tmp_dir else f"{tempfile.gettempdir()}/zomi/client",
        "ML_INSTALL_IMAGE_DIR": image_dir if image_dir else (data_dir / "images").as_posix(),
        "ML_INSTALL_CLIENT_ZM_API": zm_api,
        "ML_INSTALL_CLIENT_ZM_USER": zm_user,
        "ML_INSTALL_CLIENT_ZM_PASS": zm_pass,
        "ML_INSTALL_CLIENT_ZM_PORTAL": zm_portal,
        "ML_INSTALL_CLIENT_ROUTE_USER": mlapi_user or "imoz",
        "ML_INSTALL_CLIENT_ROUTE_PASS": mlapi_pass or "zomi",
        "ML_INSTALL_CLIENT_ROUTE_NAME": route_name or "DEFAULT FROM INSTALL <CHANGE ME!!!>",
        "ML_INSTALL_CLIENT_ROUTE_HOST": route_host or "127.0.0.1",
        "ML_INSTALL_CLIENT_ROUTE_PORT": route_port or 5000,
    }

    if args.env_file:
        parse_env_file(args.env_file)

    if args.config_only or args.secrets_only:
        secrets = False
        _cfg = False
        if args.config_only:
            _cfg = True
        if args.secrets_only:
            secrets = True

        if secrets and _cfg:
            logger.info(f"\n***Only installing config and secrets files! ***\n")
            create_("secrets", cfg_dir / "secrets.yml")
            create_("client", cfg_dir / "client.yml")
        elif secrets:
            logger.info(f"\n***Only installing secrets file! ***\n")
            create_("secrets", cfg_dir / "secrets.yml")
        elif _cfg:
            logger.info(f"\n***Only installing config file! ***\n")
            create_("client", cfg_dir / "client.yml")
        sys.exit(0)

    main()
    if THREADS:
        if any([t.is_alive() for t in THREADS.values()]):
            logger.info(
                "There is at least 1 active thread, waiting for them to finish..."
            )
            for t_name, t in THREADS.items():
                if t.is_alive():
                    logger.debug(f"Waiting for thread '{t_name}' to finish...")
                    t.join()
                    logger.debug(f"Thread '{t_name}' finished\n")
        logger.info("All threads finished")

    logger.info(f"\n\nFinished install script!!\n\n")
