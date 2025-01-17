# app/models.py

#from app import db
from app.__init__ import db

class Combination(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    numbers = db.Column(db.String(100), nullable=False) # Combinación de números en formato CSV
    special = db.Column(db.Integer, nullable=False) # Número especial

    def __repr__(self):
        return f"<Combination(id={self.id}, numbers='{self.numbers}', special={self.special})>"

    """ def __init__(self, numbers, special):
        self.numbers = ','.join(map(str, numbers))
        self.special = special """

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    numbers = db.Column(db.String(100), nullable=False)
    special = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"<PredictionHistory(id={self.id}, numbers='{self.numbers}', special={self.special})>"
    
    """ def __init__(self, numbers, special):
        self.numbers = ','.join(map(str, numbers))
        self.special = special """

    @staticmethod
    def add_prediction(prediction):
        """Método para agregar una predicción a la base de datos."""
        history = PredictionHistory(
            numbers=','.join(map(str, prediction['numbers'])),
            special=prediction['special']
        )
        db.session.add(history)
        db.session.commit()
