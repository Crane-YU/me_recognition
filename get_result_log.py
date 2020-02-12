import pickle
import pandas as pd

with open('outputs/scores_capsule_resnet_sampled_freeze.pkl', 'rb') as f:
    data = pickle.load(f)

scores = data['meter'].value()
print(scores[0])
print(scores[1])

Y_pred = data['meter'].Y_pred
Y_true = data['meter'].Y_true

me = pd.read_csv('datasets/data_apex.csv')
data = 'smic'
subject = '20'
print(data, subject)

casme2 = {}
smic = {}
samm = {}

with open('result_log.csv', 'w') as f:
    for i in range(len(Y_true)):
        y_true = Y_true[i]

        if data != me.iloc[i]['data'] or subject != me.iloc[i]['subject']:
            print(me.iloc[i]['data'], me.iloc[i]['subject'])
            data = me.iloc[i]['data']
            subject = me.iloc[i]['subject']

        log_str = str(me.iloc[i]['clip']), Y_true[i], Y_pred[i] + '\n'
        f.write(me.iloc[i]['clip'], Y_true[i], Y_pred[i])
