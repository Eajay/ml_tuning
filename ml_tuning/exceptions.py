class BasicError(Exception):
    """A basic exception"""
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None


class InvalidArgumentType(BasicError):
    """An exception raise when some arguments are not correct"""
    def __init__(self, *args):
        super().__init__(*args)

    def __str__(self):
        if self.message:
            return f'Invalid argument type, {self.message}'
        else:
            return f'Invalid argument type'


class InvalidArgumentValue(BasicError):
    """An exception raise when some arguments are not correct"""
    def __init__(self, *args):
        super().__init__(*args)

    def __str__(self):
        if self.message:
            return f'Invalid argument value, {self.message}'
        else:
            return f'Invalid argument value'
