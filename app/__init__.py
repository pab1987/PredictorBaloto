
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
from dotenv import load_dotenv
import joblib
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Importar el blueprint de rutas

# Cargar las variables de entorno
load_dotenv()

def create_app():
    
    app = Flask(__name__)
    
    # Cargar el modelo globalmente al iniciar la aplicación
    #model = joblib.load('trained_model.pkl')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///default.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Inicializar extensiones
    db.init_app(app)
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Registrar Blueprints
    from .routes import register_routes
    register_routes(app)

    return app