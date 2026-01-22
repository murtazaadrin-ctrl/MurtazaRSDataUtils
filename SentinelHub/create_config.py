from sentinelhub import SHConfig

def create_config(client_id, client_secret):
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    return config