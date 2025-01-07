## app/routes/nosotros_routes.py
from flask import Blueprint, render_template

nosotros_bp = Blueprint('nosotros', __name__)

@nosotros_bp.route('/nosotros')
def nosotros():
    return render_template('nosotros.html')