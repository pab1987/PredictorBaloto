from flask import Blueprint, render_template

user_bp = Blueprint('user', __name__)

@user_bp.route('/')
def index():
    return render_template('index.html')