from flask import Flask
import flask
from random import randint
from play import server_play 

app = Flask(__name__)
function = server_play 

@app.route("/move/<number>")
def hello_world(number):
    global function 
    result, function = function(50, int(number))
    print(result)
    response = flask.jsonify({'x': result//9, "y": result%9}) 
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/refresh")
def refresh():
    global function
    function = server_play
    response = flask.jsonify({})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
