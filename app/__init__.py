from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

# Inicializar extensiones
db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///default.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Inicializar extensiones
    db.init_app(app)
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Registrar Blueprints
    from .routes import register_routes
    register_routes(app)

    return app