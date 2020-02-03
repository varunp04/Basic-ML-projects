import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating = 4.0)

print(repr(data['train']))
print(repr(data['test']))

#model with loss func.
model = LightFM(loss = 'warp')

model.fit(data['train'],epochs = 30,num_threads = 2)

def sample_rec(model, data, user_ids):

    n_users,n_items = data['train'].shape
    for i in user_ids:

        known_pos = data['item_labels'][data['train'].tocsr()[i].indices]
        scores = model.predict(i,np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        #printing
        print("User  %s"% i)
        print(" Known positives:")
        for j in known_pos[:3]:     ##top 3
            print("     %s" % j)
        print(" Recommended:")
        for j in top_items[:3]:
            print("     %s" % j)

sample_rec(model,data,[3,45,250])

