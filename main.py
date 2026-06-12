from it.valtellina.interazioni_utente.flask_manager import FlaskManager
from it.valtellina.interazioni_utente.testing_terminale import TestingTerminale

testing = True

if testing:
    TestingTerminale.test()

else:
    # flask
    app = FlaskManager()
    app.run(host='0.0.0.0', port=5000, debug=True)