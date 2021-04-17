from flask import Flask
from flask_cors import CORS
import routes

app = Flask('ml-api')
CORS(app)
app.register_blueprint(routes.model_predict)
# Change Host Here
if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run()