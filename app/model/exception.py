class UserAlreadyExistsError(Exception):
    def __init__(self, field: str, value: str):
        super().__init__(f"{field} already exists: {value}")
        self.field = field
        self.value = value