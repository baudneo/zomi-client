import enum

class ClientEnvVars(str, enum.Enum):
    client_config_file = "ML_CLIENT_CONF_FILE"


class ServerEnvVars(str, enum.Enum):
    server_config_file = "ML_SERVER_CONF_FILE"

class ZM_ML_EnvVars(ServerEnvVars, ClientEnvVars):
    pass

