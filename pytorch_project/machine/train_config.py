from fancy import config as cfg


class TrainConfig(cfg.BaseConfig):
    train_rate: float = cfg.Option(default=0.8, type=float)
    test_rate: float = cfg.Option(default=0.2, type=float)
