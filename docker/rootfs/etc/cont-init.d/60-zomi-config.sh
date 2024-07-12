#!/command/with-contenv bash
# shellcheck shell=bash
. "/usr/local/bin/logger"
program_name="zomi-config"

TEMPLATE_SRC="/opt/zomi/client/conf"
CFG_FILE="$ML_CLIENT_CONF_FILE"
CFG_DIR=$(dirname "$ML_CLIENT_CONF_FILE")
SECRETS_FILE="$CFG_DIR"/secrets.yml

if [ ! -f "$CFG_FILE" ]; then
  echo "Config file not found, copying in template from $TEMPLATE_SRC" | info "[${program_name}] "
  cp "$TEMPLATE_SRC"/client.yml "$CFG_FILE"
  else
    echo "Config file found: $CFG_FILE" | info "[${program_name}] "
fi

if [ ! -f "$SECRETS_FILE" ]; then
  echo "Secrets file not found, copying in template from $TEMPLATE_SRC" | info "[${program_name}] "
  cp "$TEMPLATE_SRC"/secrets.yml "$SECRETS_FILE"
  else
    echo "Secrets file found: $SECRETS_FILE" | info "[${program_name}] "
fi


echo "Setting config dir \"${CFG_DIR}\" permissions for user www-data" | info "[${program_name}] "
chown -R www-data:www-data \
  "$CFG_DIR"
chmod -R 755 \
  "$CFG_DIR"
