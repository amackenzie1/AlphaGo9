from play import stateless_server_play
import json 

def lambda_handler(event, context):
    moves = list(map(int, event["queryStringParameters"]["moves"].split(",")))
    result = stateless_server_play(50, moves)
    return json.dumps({
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": {
            'x': result // 9,
            'y': result % 9
        }
    })

