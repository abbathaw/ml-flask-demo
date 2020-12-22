To get started:

```
git clone https://github.com/abbathaw/ml-flask-demo.git
$ cd ml-flask-demo

$ python3 -m venv env

// mac, linux
$ source env/bin/activate 
//windows
$ .\env\Scripts\activate.bat

$ pip3 install -r requirements.txt
```

Run locally

```
$ python3 app.py
```

Test locally

```
GET
http://localhost:5000


POST
http://localhost:5000/predict  (pass in a JSON body with your post request)


```

Deploying to Heroku

```
$ heoku login
$ heroku create
$ git push heroku main
$ heroku open
```
