class ConstantList:
    def __init__(self, value):
        self._value = value

    def __getitem__(self, index):
        return self._value
