class Config:
    _grid_size = 25

    def __new__(cls, *args, **kwargs):
        # for sigleton
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def get_grid_size():
        return Config._grid_size