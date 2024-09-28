

# ZoneMinder Machine Learning Client (ZoMi ML Client)
[!CAUTION]
> :warning: This software is in **ALPHA** stage, expect issues and incomplete, unoptimized, janky code ;) :warning:

This is a project aiming to update how [ZoneMinder](https://github.com/ZoneMinder/zoneminder) Object Detection works.
A server and client are supplied to allow for easy integration with ZoneMinder or works of software based on ZM.

## Upgrade Pip
```bash
# System wide
sudo python3 -m pip install --upgrade pip
# User
python3 -m pip install --upgrade pip
```

## Thanks

- [@pliablepixels](https://github.com/pliablepixels) for [zmNinja](https://github.com/ZoneMinder/zmNinja), [zmeventnotification](https://github.com/ZoneMinder/zmeventnotification), [mlapi](https://github.com/ZoneMinder/mlapi) and [PyZM](https://github.com/ZoneMinder/pyzm).
- [@connortechnology](https://github.com/connortechnology) for their work on [ZoneMinder](https://zoneminder.com)

# Prerequisites

- ZoneMinder 1.37.5+ (*EventStartCommand* is **REQUIRED**)
  - debian based distros can [build a .deb package for the 1.37 dev branch](https://gist.github.com/baudneo/d352c5a944a5d1371c9dfe455056e0a2)
- Python 3.8+ (3.10+ recommended) 
- Python packages required by the [install script](examples/install.py)
  - `psutil`
  - `requests`
  - `tqdm`
  - `distro`
- OpenCV (Contrib) 4.2.0+ (4.8.0+ recommended) with Python3 bindings.
  - *OpenCV (`opencv-contrib-python-headless`) is installed by default into the programs venv dir, advanced users can instead uninstall it and link in their CUDA compiled OpenCV* 

### Notes:

1. [**EventStartCommand**/**EventEndCommand**](https://zoneminder.readthedocs.io/en/latest/userguide/definemonitor.html#recording-tab:~:text=events%20are%20recorded.-,Event%20Start%20Command,the%20command%20will%20be%20the%20event%20id%20and%20the%20monitor%20id.,-Viewing%20Tab) is what runs the object detection script (Push method). Before, SHM was polled (Pull method) every \<X> seconds to see if a new event had been triggered.

# Client Info
The client uses the new ZoneMinder EventStartCommand/EventEndCommand option.
This means ZM kicks off the ML chain instead of the ML chain scanning SHM looking for an event, more efficient!

The client grabs images, sends the images to a ZoMi ML API server, filters detected responses, post processes images and sends notifications. All the heavy computation of ML models is done by the server.


## Client Pre-requisites
- Client **MUST** be installed on same host as ZM server. Multi-server ZM installs (untested) will require a client install on each server.
- `libgeos-dev` : system package (used for the Shapely python module; Polygons)

# Install
## Install Script
The [install script](examples/install.py) will install the client for you. It will also install the required python packages into its own venv.
You can supply the script with options to configure the install. Below is a list of the options and their defaults.

## Install Script Options

- `--debug` : Enable debug logging
- 
