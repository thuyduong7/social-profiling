# %%
from datetime import datetime
import uuid
import os

run_uuid = uuid.uuid4().hex
start_time = str(datetime.now().isoformat())

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

p_use_db = False
p_purge_db = False
CONNECTION_STRING = "mongodb://admin:admin@localhost:27018/?authSource=admin"
p_db = 'facial_indexer'
p_face_collection = 'embeddings'
p_profile_collection = 'profiles'

p_do_split_dataset = False
p_dataset_ratio = [0.8, 0.2]
p_split_seed = 6969
orig_path = '../tests/VN-celeb/'
split_path = '../tests/VN-celeb-split/'
train_path = os.path.join(split_path, 'train')
val_path = os.path.join(split_path, 'val')

# Parameters
p_models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
p_model = "Facenet"
CNN_OUTPUT_SIZE = 128
CNN_INPUT_SIZE = (160, 160)

p_backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'skip']
p_detector_backend = p_backends[-1]

p_grayscale = False
p_enforce_detection = False

p_metrics = ['euclidean', 'cosine']
p_metric = 'cosine'
p_normalize_vectors = True
p_cuda_index = False

p_consensus_test_upper_inclusive = 32

p_export_folder = f'../results/{run_uuid}/'
p_export_json_class2pid = os.path.join(p_export_folder, f'train_class2pid.json')
p_export_json_train_faces = os.path.join(p_export_folder, f'train_faces.json')
p_export_json_train_profiles = os.path.join(p_export_folder, f'train_profiles.json')
p_export_json_val_faces = os.path.join(p_export_folder, f'val_faces.json')
p_export_json_faiss_kneighbors = os.path.join(p_export_folder, f'val_faiss_k32_neighbors.json')
p_export_bin_faiss_index = os.path.join(p_export_folder, f'faiss_index.bin')

p_result_eval = os.path.join(p_export_folder, f'eval.csv')
p_result_accuracy = os.path.join(p_export_folder, f'accuracy.csv')

p_tensorflow_export = os.path.join(p_export_folder, f'{p_model}_{start_time.replace(":", "").replace(".", "")}')

# %% [markdown]
# ### Dataset Preparation: 80-20 split

# %% [markdown]
# Using `splitfolders` to split dataset into training and validation set with customizable ratio and seed
# 
# Here, 80-20 is chosen

# %%
import splitfolders

if p_do_split_dataset:
    splitfolders.ratio(
        orig_path, output=split_path, seed=p_split_seed, ratio=(p_dataset_ratio[0], p_dataset_ratio[1])
    )

# %% [markdown]
# ### Train

# %% [markdown]
# #### Facial Embedding vectors and Database insertions

# %% [markdown]
# Gather image path and group by class (profile) into a `dict{type, name, images}`.

# %%
base_path = train_path

profiles = []

for root, dirs, files in os.walk(base_path):
    if not dirs:
        person_name = root.split('/')[-1]
        img_paths = []

        for file in files:
            img_path = root + '/' + file
            img_paths.append(img_path)
        
        # filenames[person_name] = img_paths
        profile = {
            'type': 'profile',
            'name': person_name,
            'images': img_paths
        }
        profiles.append(profile)

# %% [markdown]
# `get_face_embedding()`: takes in an image, return a corresponding vector representation (facial embedding)
# 
# `process_profile()`: takes in a ProfileDict (mentioned above), and convert all image paths into their respective facial embeddings

# %%
from copy import deepcopy
from deepface.basemodels import Facenet
from deepface.commons import functions
from deepface import DeepFace

# model = Facenet.loadModel()
model = DeepFace.build_model(p_model)

def get_face_embedding(
                            img, grayscale = False, enforce_detection = True, 
                            detector_backend = 'opencv', return_region = False, align = True):
        
        target, region = functions.preprocess_face(
            img=img, target_size=CNN_INPUT_SIZE, grayscale=grayscale, enforce_detection=enforce_detection,
            detector_backend=detector_backend, return_region=True, align=align)
        target = model.predict(target)[0,:]

        if return_region == True:
            return target, region 
        else:
            return target

def process_profile(subject, grayscale = False, enforce_detection = True, detector_backend = 'opencv'):
        # Deep copy of subject
        profile = deepcopy(subject)

        # Pop the images list consisting of paths to the images
        images = profile.pop('images', None)
        # Pop the '_id' just to be sure
        if '_id' in profile:
            profile.pop('_id')
        # Add type 'profile'
        profile['type'] = 'profile'

        # Empty array storing facial embeddings document,
        # each embedding is a document, with a list w/ len=CNN_OUTPUT_SIZE stored in 'values'
        facial_embeddings = []

        # For each image path
        for image in images:
            # Process then get embedding for the face
            embedding, region = get_face_embedding(
                img=image, grayscale=grayscale, 
                enforce_detection=enforce_detection, 
                detector_backend=detector_backend,
                return_region=True)
            # append to array
            facial_embeddings.append(
                {
                    'type': 'embedding',
                    'image_path': image,
                    'values': embedding.tolist(),
                    'region': region
                }
            )

        # If no image is found, raise Exception
        if not facial_embeddings:
            raise ValueError(profile['name'])
        
        # profile['_facial_embeddings'] = facial_embeddings

        return profile, facial_embeddings

# %% [markdown]
# Initialize connection to MongoDb

# %%
if p_use_db == True:
    from pymongo import MongoClient
    import pymongo
    import gridfs

    client = MongoClient(CONNECTION_STRING)
    db = client['facial_indexer']

    profile_collection = db['profiles']
    face_collection = db['embeddings']

    fs = gridfs.GridFS(db)

    if p_purge_db:
        profile_collection.delete_many({})
        face_collection.delete_many({})

# %%
profile_counter = 0   
face_counter = 0

print("profile_counter = {}\nface_counter = {}".format(profile_counter, face_counter))

# %% [markdown]
# Process the profiles and insert them and their facial embeddings to database, also returning the inserted profiles & embeddings for further testing

# %%
from tqdm import tqdm

def add_profiles_bulk(profiles_array):
    global profile_counter
    global face_counter

    faces_map = {}
    profiles_map = {}

    for profile in tqdm(profiles_array):
        processed_profile, processed_embeddings = process_profile(profile, grayscale=p_grayscale, enforce_detection=p_enforce_detection, detector_backend=p_detector_backend)
        
        pid = f'profile_{profile_counter}'
        processed_profile['_id'] = pid
        profile_counter += 1

        profiles_map[pid] = (processed_profile)

        for fe in processed_embeddings: 
            fid = f'face_{face_counter}'
            fe['_id'] = fid
            face_counter += 1
            fe['profile_id'] = pid
            
            # fe_img_path = fe['image_path']
            # img_fsid = fs.put( open(fe_img_path, 'r') )
            # fe['image_fsid'] = img_fsid

            faces_map[fid] = (fe)        

    if p_use_db == True:
        dbq_profile = list(profiles_map.values())
        dbq_face = list(faces_map.values())
        # with client.start_session() as session:
        #     with session.start_transaction():
        try:
            profile_collection.insert_many(
                documents=dbq_profile,
                # session=session
            )
            face_collection.insert_many(
                documents=dbq_face,
                # session=session
            )
        except pymongo.errors.ServerSelectionTimeoutError:
            print("Unable to connect to database, insertion retry possible using return values from function")
            pass
        
        print(f'Inserted {len(dbq_profile)} profiles and {len(dbq_face)} faces to db')

    return faces_map, profiles_map

# %%
faces_out, profiles_out = add_profiles_bulk(profiles)

# %%
print(f'Faces: {sys.getsizeof(faces_out)}B\nProfiles: {sys.getsizeof(profiles_out)}B')

# %%
classname2pid = {}
for profile in list(profiles_out.values()): 
    classname2pid[profile['name']] = profile['_id']

# %% [markdown]
# #### FAISS indexing

# %% [markdown]
# ##### Data Prep
# 
# Juggling data into shapes compatible with FAISS insert function

# %%
import numpy as np

ids = []
embeddings = []

for face in list(faces_out.values()):
    fid = face['_id']
    # id is in format embedding_xxxxx where xxxxx is number
    fid = fid.split('_')[-1]
    embedding = face['values']

    ids.append(fid)
    embeddings.append(embedding)

ids = np.array(ids, dtype='int64')
embeddings = np.array(embeddings, dtype='f')

# %%
print(f"ids: {ids.shape}", f"embeddings: {embeddings.shape}")

# %% [markdown]
# ##### Index Init

# %%
import faiss 

if p_metric == 'euclidean':
    index_flat = faiss.IndexFlatL2(CNN_OUTPUT_SIZE)
elif p_metric == 'cosine':
    index_flat = faiss.IndexFlatIP(CNN_OUTPUT_SIZE)

index_flat = faiss.IndexIDMap(index_flat)

if p_normalize_vectors == True:
    faiss.normalize_L2(embeddings)

if p_cuda_index == True:
    # Declares a GPU resource, encapsulating a chunk of GPU memory
    res = faiss.StandardGpuResources()
    index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

# %%
index_flat.add_with_ids(embeddings, ids)

# %%
index_flat.ntotal

# %% [markdown]
# ### Evaluation

# %% [markdown]
# Creates empty dictionary with image paths as keys for later assignment with facial embeddings

# %%
base_path = val_path

paths = []

for root, dirs, files in os.walk(base_path):
    if not dirs:
        person_name = root.split('/')[-1]

        for file in files:
            img_path = root + '/' + file
            paths.append(img_path)

# %% [markdown]
# Gathering facial embeddings for all validation dataset images to cache.

# %%
import numpy as np

val_emb = {}

for path in tqdm(paths):
    embedding = get_face_embedding(path, enforce_detection = False, detector_backend = 'skip')
    val_emb[path] = embedding.tolist()

# %%
len(val_emb)

# %% [markdown]
# #### Testing face query with FAISS index

# %% [markdown]
# ##### Get k-nearest neighbors of every faces in validation dataset for later analysis and optimization

# %%
import numpy as np

val_faiss_k_neighbors = {}

for key, embedding in tqdm( list(val_emb.items()) ):
    target = np.array(embedding, dtype='f') 
    target = np.expand_dims(target, axis=0)
    if p_normalize_vectors == True:
        faiss.normalize_L2(target)
        
    distances, neighbors = index_flat.search(target, p_consensus_test_upper_inclusive)

    # matches = face_collection.find(
    #     { "_id": { "$in": [f"face_{idnum}" for idnum in neighbors[0].tolist()] } },
    #     { "profile_id": 1 }
    # )
    # matches = list(matches) 

    matches = [faces_out[f'face_{idnum}'] for idnum in neighbors[0].tolist()]

    results = [ [ np.float64(distance), match['_id'], match['profile_id'] ] for match, distance in zip(matches, distances[0]) ]

    spl = key.split('/')
    short_key = f'val/{spl[-2]}/{spl[-1]}'
    val_faiss_k_neighbors[short_key] = results

# %% [markdown]
# ##### Create result tracking dataframe

# %%
# paths2 = ['val/{}/{}'.format(path.split('/')[-2], path.split('/')[-1]) for path in paths]
val_paths2 = []
val_classnames = []
val_truth_pid = []
for path in paths: 
    spl = path.split('/')
    val_paths2.append('val/{}/{}'.format(spl[-2], spl[-1]))
    val_classnames.append(spl[-2])
    val_truth_pid.append( classname2pid[spl[-2]] )

# %%
import pandas as pd
eval_df = pd.DataFrame(
    { 
        "truth_class": val_classnames,
        "truth_pid": val_truth_pid
    },
    index=val_paths2
)

# %%
for i in range( 1, p_consensus_test_upper_inclusive + 1 ):
    eval_df[f'predict_{i}'] = np.nan 
    eval_df[f'positive_{i}'] = np.nan

# %%
eval_df.head()

# %% [markdown]
# #### Get Consensus for k=[1,25]

# %%
from statistics import mode 
def get_consensus(result_arr, top_k=[ None ]):
    distances, embeddings, profiles = list( zip(*result_arr) )
    
    ret = ()
    for k in top_k:
        ret = ret + ( mode(profiles[0:k]), )
    return ret

# %%
for key, value in val_faiss_k_neighbors.items():
    predictions = get_consensus( value,  list(range( 1, p_consensus_test_upper_inclusive + 1 )) )
    eval_df.at[key, 'predicted'] = True
    for n in range( 1, p_consensus_test_upper_inclusive + 1 ):
        eval_df.at[key, f'predict_{n}'] = predictions[n - 1]

# %%
for n in range( 1, p_consensus_test_upper_inclusive + 1 ):
    eval_df[f'positive_{n}'] = np.where( (eval_df['truth_pid'] == eval_df[f'predict_{n}']), True, False )

# %%
eval_df[ eval_df['predicted'] == True ].head()

# %%
total_count = eval_df[ eval_df['predicted'] == True ].shape[0]
total_count

# %%
accuracy_series = []
positive_series = []

for n in range( 1, p_consensus_test_upper_inclusive + 1 ):
    k_positive_count = eval_df[ eval_df['truth_pid'] == eval_df[f'predict_{n}'] ].shape[0]
    accuracy = k_positive_count/total_count
    positive_series.append(k_positive_count)
    accuracy_series.append(accuracy)
    print(n, k_positive_count, accuracy)

# %%
result_df = pd.DataFrame(
    { 
        "positive": positive_series,
        "accuracy": accuracy_series,
    },
    index=list(range( 1, p_consensus_test_upper_inclusive + 1 ))
)
result_df

# %%
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

sns.lineplot(data=result_df, y='accuracy', x=result_df.index)

# %% [markdown]
# ### Save Results

# %%
if not os.path.exists(p_export_folder):
    os.makedirs(p_export_folder)

# %%
eval_df.to_csv(p_result_eval)
result_df.to_csv(p_result_accuracy)

# %%
model.save(p_tensorflow_export)

# %%
import json

with open(p_export_json_train_faces, 'w') as f:
    json.dump(faces_out, f)

with open(p_export_json_train_profiles, 'w') as f:
    json.dump(profiles_out, f)

print(f'Model output written. {len(faces_out)} and {len(profiles_out)} dictionary keys')

# %%
with open(p_export_json_class2pid, 'w') as f:
    json.dump(classname2pid, f)

print("Classname-ProfileId map written to disk")

# %%
with open(p_export_json_val_faces, 'w') as f:
    json.dump(val_emb, f)

print("Validation Face Embeddings: Written {} {}-length vectors".format(len(val_emb), len(embedding)))

# %%
with open(p_export_json_faiss_kneighbors, 'w') as f:
    json.dump(val_faiss_k_neighbors, f)

print(f"Written {len(val_faiss_k_neighbors)} dicts, each with {p_consensus_test_upper_inclusive}x{CNN_OUTPUT_SIZE}-length vectors")

# %%
if p_cuda_index == True:
    index_flat_cpu = faiss.index_gpu_to_cpu(index_flat)
    faiss.write_index(index_flat_cpu, p_export_bin_faiss_index)
else:
    faiss.write_index(index_flat, p_export_bin_faiss_index)

# %%
end_time = str(datetime.now().isoformat())

run_settings = {
    'id': run_uuid,
    'start': start_time,
    'end': end_time,
    'database': {
        'use_database': p_use_db,
        'purge_database': p_purge_db,
        'connection_string': CONNECTION_STRING,
        'database_name': p_db,
        'profiles_collection': p_profile_collection,
        'embeddings_collection': p_face_collection,
    },
    'dataset': {
        'do_split_dataset': p_do_split_dataset,
        'dataset_ratio': p_dataset_ratio,
        'dataset_split_seed': p_split_seed,
        'dataset_paths': {
            'original': base_path,
            'split_path': split_path,
            'train_path': train_path,
            'val_path': val_path,
        },
    },
    'face_model': p_model,
    'face_detector_backend': p_detector_backend,
    'face_grayscale': p_grayscale,
    'face_enforce_detection': p_enforce_detection,
    'faiss_metric': p_metric,
    'faiss_normalize_vectors': p_normalize_vectors,
    'faiss_use_cuda': p_cuda_index,
    'consensus_test_range': [1, p_consensus_test_upper_inclusive],
    'exports': {
        'root': p_export_folder,
        'class2pid': p_export_json_class2pid,
        'train_faces': p_export_json_train_faces,
        'train_profiles': p_export_json_train_profiles,
        'val_faces': p_export_json_val_faces,
        'faiss_query_results': p_export_json_faiss_kneighbors,
        'faiss_index': p_export_bin_faiss_index,
        'results_compare_table': p_result_eval,
        'results_accuracy_table': p_result_accuracy,
        'tensorflow_export': p_tensorflow_export
    }
}

# %%
with open( os.path.join(p_export_folder, 'params.json'), 'w' ) as f:
    json.dump(run_settings, f)

print("Written parameters to file")

# %%
print(run_settings)

# %%
print(f"Test Run {run_uuid} completed. From {start_time} to {end_time}. Results and Parameters stored at {p_export_folder}")

# %%
################################################################################################
################################################################################################
################################################################################################
################################################################################################


