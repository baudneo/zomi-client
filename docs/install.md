# Make sure to edit the config file after installation

This script will install into its own venv and create the necessary directories and files (default: `/opt/zomi/client`). 
It will modify the system-wide python script with a shebang (`#!`) pointing to its venv.

# CLI Arguments

You can always supply `--help` to the script to get a breakdown.

## Install script configuration
`--user` and `--group` will attempt to find the user and group that the HTTP daemon is running as. 

If more than 1 user is present, it will use the highest UID. If more than 1 group is present, 
it will use the highest GID. This is to avoid using root as the install user.

| Argument                      | Description                                       | Default                          |
|-------------------------------|---------------------------------------------------|----------------------------------|
| `--user`                      | User to install as                                | Automatic HTTP daemon user       |
| `--group`                     | Group to install as                               | Automatic HTTP daemon group      |
| `-D / --debug`                | Enable debug logging                              | `False`                          |
| `-T  / --test / --dry-run`    | Test mode (don't install anything)                | `False`                          |
| `-I / --interactive`          | Interactive mode                                  | `False`                          |
| `--system-create-permissions` | System octal [0o] file permissions                | `0o755`                          |
| `--config-create-permissions` | Config octal [0o] file permissions                | `0o755`                          |
| `--no-cache-dir`              | Disable pip cache directory                       | `False`                          |
| `--editable`                  | Install in pip editable mode                      | `False`                          |
| `--config-only`               | Only install config files (use with secrets-only) | `False`                          |
| `--secrets-only`              | Only install secrets files (use with config-only) | `False`                          |
| `--install-log`               | Log file for installation                         | `./zomi-client_install.log`      |
| `--dir-data`                  | Directory for ZoMi data files                     | `/opt/zomi/client`               |
| `--dir-config`                | Directory for ZoMi config files                   | `/{data dir}/conf`               |
| `--dir-models`                | Directory for ZoMi model files                    | `/{data dir}/models`             |
| `--dir-images`                | Directory for ZoMi image files                    | `/{data dir}/images`             |
| `--dir-temp`                  | Directory for ZoMi temporary files                | `/{system temp dir}/zomi/client` |
| `--dir-log`                   | Directory for ZoMi log files                      | `/{data dir}/logs`               |

## Config file helpers
These are *optional* 'helper' arguments that will help fill out some of the templated config file.

### ZoMi ML API configuration

| Argument                      | Description                        | Default                        |
|-------------------------------|------------------------------------|--------------------------------|
| `--mlapi-user`                | ZoMi ML API username               | imoz                           |
| `--mlapi-pass`                | ZoMi ML API password               | zomi                           |
| `--mlapi-url`                 | ZoMi ML API URL                    | http://localhost               |
| `--mlapi-port`                | ZoMi ML API port                   | 5000                           |
| `--mlapi-name`                | ZoMi ML API name                   | Default Route                  |

### ZoneMinder configuration
| Argument      | Description     | Default             |
|---------------|-----------------|---------------------|
| `--zm-portal` | ZM Portal URL   | `None`              |
| `--zm-api`    | ZM API URL      | `/{portal url}/api` |
| `--zm-user`   | ZM API username | `None`              |
| `--zm-pass`   | ZM API password | `None`              |

# System wide files installed
2 files are installed into `/usr/bin`:
- `zomi-ESC` shell script - the ZM `EventStartCommand` script
- `zomi-eventproc` python script that `zomi-ESC` calls

# Example
Start in the repo base directory
```bash
# Do a test run with debug logging and zm config helpers
./examples/install.py -D -T --zm-portal http://zm.example.com/zm --zm-user zmapiuser --zm-pass zmapipass

# Install as a specific user and group with zm config helpers
./examples/install.py --user www-data --group www-data --zm-portal http://zm.example.com --zm-user zmapiuser --zm-pass zmapipass 

# This should 'just work' with the defaults, interactive mode.
./examples/install.py -I

```

:warning: Edit the config file after installation :warning: