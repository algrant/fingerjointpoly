import bottle
import fjp_crud
import allow_cors

# mostly cribbed from https://www.toptal.com/bottle/building-a-rest-api-with-bottle-framework
# if cors attacks this has a theoretical answer

app = application = bottle.default_app()

if __name__ == '__main__':
    bottle.run(host = '127.0.0.1', port = 8000, reloader=True)