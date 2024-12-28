from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
import os

db = SQLAlchemy()

def create_app():
    load_dotenv()
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///default.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    CORS(app)

    with app.app_context():
        db.create_all()

    # Importar y registrar los blueprints
    from app.routes.admin import admin_bp
    from app.routes.user import user_bp
    from app.routes.api import api_bp
    from app.routes.predictions import predictions_bp

    app.register_blueprint(admin_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(predictions_bp)

    return app
