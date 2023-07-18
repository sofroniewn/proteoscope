import hydra
from hydra.core.config_store import ConfigStore
from proteoscope import ProteoscopeConfig, train_cytoself, train_proteoscope


cs = ConfigStore.instance()
cs.store(name="rosa_config", node=ProteoscopeConfig)

CONFIG_PATH = "../conf"
CONFIG_NAME = "config"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(config: ProteoscopeConfig) -> None:
    import os

    print("Working directory : {}".format(os.getcwd()))
    train_cytoself(config)


if __name__ == "__main__":
    main()
