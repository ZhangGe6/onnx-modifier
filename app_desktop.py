from flaskwebgui import FlaskUI
from app import app

FlaskUI(app, width=1200, height=800).run()