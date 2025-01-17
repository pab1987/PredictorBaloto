from .views_routes import views_bp
from .api_routes import api_bp

def register_routes(app):
    app.register_blueprint(views_bp) # Rutas para vistas HTML
    app.register_blueprint(api_bp, url_prefix='/api') # Rutas para API