#####################################################################
#                                                                   #
# Convert rootfs to LF using dos2unix                               #
# Alleviates issues when git uses CRLF on Windows                   #
#                                                                   #
#####################################################################
FROM alpine:latest as rootfs-converter
WORKDIR /rootfs

RUN set -x \
    && apk add --repository=http://dl-cdn.alpinelinux.org/alpine/edge/community/ \
        dos2unix

COPY rootfs .
RUN set -x \
    && find . -type f -print0 | xargs -0 -n 1 -P 4 dos2unix \
    && chmod -R +x *

#####################################################################
#                                                                   #
# Install ZoMi Client                                               #
#                                                                   #
#####################################################################
FROM ghcr.io/zoneminder-containers/zoneminder-base:latest as zm-base

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt

# Install runtime dependencies
RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      python3 python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install zomi-client
RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      git python3-pip python3-setuptools python3-wheel libgeos-dev \
    && pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir psutil distro tqdm requests \
    && git clone https://github.com/baudneo/zomi-client.git zomi-src \
    && cd zomi-src \
    && python3 examples/install.py \
    && apt-get remove -y git python3-pip python3-setuptools python3-wheel libgeos-dev \
    && apt-get clean \
#    && rm -rf /opt/zomi-src \
    && rm -rf /var/lib/apt/lists/*

ENV ML_CLIENT_CONF_FILE=/opt/zomi-config/client.yml

# Copy rootfs
COPY --from=rootfs-converter /rootfs/etc/s6-overlay/scripts /etc/s6-overlay/scripts
COPY --from=rootfs-converter /rootfs/etc/s6-overlay/cont.init.d /etc/s6-overlay/cont.init.d
