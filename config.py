import json


class CameraConfig:
    def __init__(self, filename):
        """setting default parameter
        """
        super(CameraConfig, self).__init__()
        try:
            with open(filename) as f:
                data = json.load(f)
            self.camera = data["cameraName"]
            self.sensor_width = data['cameraSensorWidth']
            self.sensor_height = data['cameraSensorHeight']
            self.Icx = data['iCx']
            self.Icy = data['iCy']
            self.pp = (self.Icx, self.Icy)
            self.ratio = data['ratio']
            self.imageWidth = data['imageWidth']
            self.imageHeight = data['imageHeight']
            self.calibrationRatio = data['calibrationRatio']
            self.parameter0 = data['parameter0']
            self.parameter1 = data['parameter1']
            self.parameter2 = data['parameter2']
            self.parameter3 = data['parameter3']
            self.parameter4 = data['parameter4']
            self.parameter5 = data['parameter5']

        except OSError as err:
            print("OS error: {0}".format(err))
