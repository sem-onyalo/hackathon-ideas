import json

class App:
    _settingsFileName = 'app.settings.json'

    def getSettings(self, name):
        with open(self._settingsFileName, 'r') as fh:
            settings = json.loads(fh.read())
            if name in settings:
                return settings[name]

        raise RuntimeError(f'Name {name} not in settings file')