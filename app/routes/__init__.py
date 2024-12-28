## app/routes/__init__.py
def register_routes(app):
    from .user_routes import user_bp
    from .admin_routes import admin_bp
    from .api_routes import api_bp
    from .prediction_routes import prediction_bp

    app.register_blueprint(user_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(prediction_bp)