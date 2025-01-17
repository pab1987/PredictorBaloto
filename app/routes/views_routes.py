from flask import Blueprint, render_template

views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def home():
    return render_template('index.html')

@views_bp.route('/admin')
def admin():
    return render_template('admin.html')

@views_bp.route('/nosotros')
def nosotros():
    return render_template('nosotros.html')
