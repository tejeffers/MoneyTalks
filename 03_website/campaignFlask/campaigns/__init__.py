from flask import Flask
app = Flask(__name__)
from campaigns import views
