from flask import Flask, render_template, request
import sqlite3, random
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app=Flask(__name__); DB="zeroplate.db"

def init_db():
    c=sqlite3.connect(DB)
    c.execute('''CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, meal TEXT,
    attendance INTEGER, previous INTEGER, avg7 INTEGER, event INTEGER,
    predicted REAL, recommended INTEGER)''')
    c.commit(); c.close()

def train():
    random.seed(42); rows=[]
    for _ in range(500):
        a=random.randint(80,700); p=int(a*random.uniform(.82,1.02))
        av=int(a*random.uniform(.84,1.00)); e=random.randint(0,1)
        d=.45*a+.28*p+.25*av+18*e+random.gauss(0,15)
        rows.append([a,p,av,e,max(1,int(d))])
    df=pd.DataFrame(rows,columns=["attendance","previous","avg7","event","demand"])
    m=RandomForestRegressor(n_estimators=100,random_state=42)
    m.fit(df[["attendance","previous","avg7","event"]],df.demand)
    return m

model=train(); init_db()

@app.route("/")
def home(): return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    meal=request.form["meal"]; a=int(request.form["attendance"])
    p=int(request.form["previous"]); av=int(request.form["avg7"])
    e=1 if request.form["event"]=="yes" else 0
    pred=float(model.predict([[a,p,av,e]])[0]); rec=round(pred*1.03)
    c=sqlite3.connect(DB); c.execute(
      "INSERT INTO predictions(created_at,meal,attendance,previous,avg7,event,predicted,recommended) VALUES(?,?,?,?,?,?,?,?)",
      (datetime.now().strftime("%Y-%m-%d %H:%M"),meal,a,p,av,e,pred,rec))
    c.commit(); c.close()
    return render_template("result.html",meal=meal,attendance=a,predicted=round(pred),recommended=rec)

@app.route("/dashboard")
def dashboard():
    c=sqlite3.connect(DB); df=pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC",c); c.close()
    return render_template("dashboard.html",rows=df.to_dict("records"),count=len(df),
                           avg=round(df.predicted.mean()) if len(df) else 0)

if __name__=="__main__": app.run(debug=True)
