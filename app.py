from flask import Flask, render_template, request, url_for, redirect
import pickle 
import numpy as np

app= Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Predict', methods=['GET','POST'])
def Predict():
    team1 = str(request.args.get('list1'))
    team2 = str(request.args.get('list2'))

    toss_win = int(request.args.get('toss_winner'))
    choose = int(request.args.get('fb'))

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('inv_vocab.pkl', 'rb') as f:
        inv_vocab = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    cteam1 = inv_vocab[team1]
    cteam2 = inv_vocab[team2]

    if cteam1 == cteam2:
        return redirect(url_for('index'))

    lst = np.array([cteam1, cteam2, choose, toss_win], dtype='int').reshape(1,-1)

    prediction = model.predict(lst)
    if prediction == 0:
        team_win = team1

    else:
        team_win = team2


    return render_template('Predict.html', data=team_win)

if __name__ == "__main__":
    app.run(debug=True)
