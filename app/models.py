# app/models.py

from . import db

class Combination(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    numbers = db.Column(db.String(100), nullable=False)
    special = db.Column(db.Integer, nullable=False)

    def __init__(self, numbers, special):
        self.numbers = ','.join(map(str, numbers))
        self.special = special

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    numbers = db.Column(db.String(100), nullable=False)
    special = db.Column(db.Integer, nullable=False)

    def __init__(self, numbers, special):
        self.numbers = ','.join(map(str, numbers))
        self.special = special

    @staticmethod
    def add_prediction(prediction):
        history = PredictionHistory(
            numbers=','.join(map(str, prediction['numbers'])),
            special=prediction['special']
        )
        db.session.add(history)
        db.session.commit()
