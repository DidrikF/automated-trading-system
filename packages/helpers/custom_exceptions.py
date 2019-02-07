class FeatureError(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...

        self.errors = errors
    
    def __str__(self):
        return super().__str__() + ": " + str(self.errors)