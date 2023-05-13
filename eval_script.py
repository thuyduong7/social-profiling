# Evaluation Script

# %% [markdown]
# # Retrieving facial embeddings and building the index

# %%
from pymongo import MongoClient
import pymongo

CONNECTION_STRING = "mongodb://admin:admin@localhost:27018/?authSource=admin"

client = MongoClient(CONNECTION_STRING)
db = client['social_profiling']

profile_collection = db['eval_profiles']
face_collection = db['eval_embeddings']
# %%
faces = face_collection.find()
faces = list(faces)

print(f"[INFO] Imported {len(faces)} faces from {db}")
# %%
import numpy as np

ids = []
embeddings = []
for face in faces:
    fid = face['_id']
    fid = fid.split('_')[-1]
    embedding = face['values']
    ids.append(fid)
    embeddings.append(embedding)

ids = np.array(ids, dtype='int64')
embeddings = np.array(embeddings, dtype='f')

# %%
print(f"[INFO] IDs array shape: {ids.shape}")
print(f"[INFO] Embeddings array shape: {embeddings.shape}")

# %%
import faiss

dimensions = 128    # FaceNet output is 128d vector

metric = 'euclidean' #euclidean, cosine
 
if metric == 'euclidean':
    index = faiss.IndexFlatL2(dimensions)
elif metric == 'cosine':
    index = faiss.IndexFlatIP(dimensions)
    faiss.normalize_L2(embeddings)

index = faiss.IndexIDMap(index)

print(f"[INFO] Initialized FAISS index")
# %%
index.add_with_ids(embeddings, ids)
print(f"[INFO] Embeddings and IDs added to FAISS index")
# %%
index_bin_path = "../eval.index"
faiss.write_index(index, "../eval.index")
print(f"[INFO] FAISS index written to {index_bin_path}")
# %%
split_path = './tests/nguoinoitiengtv/split/'
import os
train_path = os.path.join(split_path, 'train')
val_path = os.path.join(split_path, 'val')
print(f"[INFO] Working dir --> {val_path}")
# %%
import json
f2p_map_path = os.path.join(split_path, 'filename_profile_map.json')
f2p_map = []

with open( f2p_map_path, 'r' ) as f:
    f2p_map = json.load(f)
print(f"[INFO] Loaded filename to profile id map at {f2p_map_path}")
# %%
import pandas as pd

df = pd.DataFrame.from_dict(f2p_map, orient='index')

df.columns = ['truth']
df['is_in_val'] = np.nan
df['has_detect_face'] = np.nan
df['prediction'] = np.nan

print(f"[INFO] Created Results tracking df", df)

# %%
import os

base_path = val_path

val_filenames = []

for root, dirs, files in os.walk(base_path):
    if not dirs:
        for f in files:
            path = os.path.join(root, f)
            val_filenames.append(path)
            
len(val_filenames)
print(f"[INFO] Imported working file paths. Total: {len(val_filenames)}")
# %%
def get_id(prefix, suffix, rjust_fill=6):
    return (str(prefix) + str(suffix).rjust(rjust_fill, '0'))

# %%
from deepface.basemodels import Facenet
from deepface.commons import functions

model = Facenet.loadModel()
print("[INFO] Initialized Facenet model")

def get_closest_match(target_img_path, k, index):
    target_img = functions.preprocess_face(img=target_img_path, target_size=(160, 160), detector_backend='mtcnn')
    target_representation = model.predict(target_img)[0,:]

    target_representation = np.array(target_representation, dtype='f')
    target_representation = np.expand_dims(target_representation, axis=0)

    distances, neighbors = index.search(target_representation, k)

    return distances, neighbors

# %%
from tqdm import tqdm
import numpy as np
import time

test_result = {}

print("[START] Beginning test procedure")
# Begin Process
tic = time.time()

for path in tqdm(val_filenames):
    head, tail = os.path.split(path)
    df.at[tail, 'is_in_val'] = True

    # try:
    #     preprocessed_faces[tail] = functions.preprocess_face(img=path, target_size=(160, 160))
    # except:
    #     df.at[tail, 'has_detect_face'] = False

    # # print(path)
    # # d, n = get_closest_match(path, 1, index)
    # # print(path, d, n)
    try:
        d, n = get_closest_match(path, 20, index)
    except ValueError:
        df.at[tail, 'has_detect_face'] = False
        continue

    matches = face_collection.find(
        { "_id": { "$in": [get_id("embedding_", idnum) for idnum in n[0].tolist()] } },
        { "profile_id": 1 }
        )
    matches = list(matches)

    results = [ [ match['_id'], match['profile_id'], np.float64(distance) ] for match, distance in zip(matches, d[0]) ]

    test_result[tail] = results
            
    # match = face_collection.find_one({ "_id": get_id('embedding_', n[0][0]) }, { "profile_id": 1 })
    # profile_id = match['profile_id']

    # print(f"{d} {n} {profile_id}")
    
    # df.at[tail, 'prediction'] = profile_id
    df.at[tail, 'has_detect_face'] = True

toc = time.time()
print(f"[END] Time elapsed = {toc - tic}s")

import json
test_result_path = os.path.join(split_path, 'test_result.json')
with open(test_result_path, 'w') as f:
    json.dump(test_result, f)
print(f"[INFO] Written test result to {test_result_path}")
# # %%
# with open(os.path.join(split_path, 'test_result.json'), 'r') as f:
#     import_result = json.load(f)

# # %%
# import_result

# # %%
# df[ df['has_detect_face'] == True ]

# %%
test_table_path = os.path.join(split_path, 'test_result.csv')
df.to_csv(test_table_path)
print(f"[INFO] Written Results tracking table to {test_table_path}")

# # %%
# df[ df['truth'] == df['prediction'] ]

# # %%
# df[ df['is_in_val'] == True ]


