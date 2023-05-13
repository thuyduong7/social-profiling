import seaborn as sns
import matplotlib.pyplot as plt
from readline import read_init_file
from statistics import mode
import faiss
from facialindex.error import FacialIndexError, ImagePreprocessFaceError
from tqdm import tqdm
import numpy as np
import sys
from facialindex.facialindex import FacialIndex
from copy import deepcopy
import json
import pandas as pd
import splitfolders
import os
import uuid
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(
    description="Using FacialIndex class to test dataset")

parser.add_argument(
    '--usedb', type=int, choices=[0, 1], required=True, help='Use mongoDB or not'
)

parser.add_argument(
    '--mongodb', type=str, help='mongodb connection string'
)

parser.add_argument(
    '--dbname', type=str, help='mongodb database name'
)

parser.add_argument(
    '--resume', type=int, choices=[0, 1], required=True, help='skip add profiles phase, construct index from database'
)

parser.add_argument(
    '--datasetsplit', type=int, choices=[0, 1], required=True, help='trigger datasetsplits'
)

parser.add_argument(
    '--datasetseed', type=int, help='splitfolders seed'
)

parser.add_argument(
    '--similaritymeasure', type=str, choices=['cosine', 'euclidean'], required=True, help='faiss index type'
)

parser.add_argument(
    '--normalizevectors', type=int, choices=[0, 1], required=True, help='normalizeL2 vectors'
)

parser.add_argument(
    '--enforcefacedetection', type=int, choices=[0, 1], required=True, help='enforce face detection'
)

parser.add_argument(
    '--facevectormodel', type=str,
    choices=["VGG-Face", "Facenet", "Facenet512",
             "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"],
    required=True, help='face vectorization model'
)

parser.add_argument(
    '--facedetectbackend', type=str,
    choices=['opencv', 'ssd', 'dlib',
             'mtcnn', 'retinaface', 'mediapipe', 'skip'],
    required=True, help='face vectorization model'
)

args = parser.parse_args()

print(args)

input("Verify Arguments")

# %%

run_uuid = uuid.uuid4().hex
start_time = str(datetime.now().isoformat())

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

p_use_db = True if args.usedb == 1 else False
p_purge_db = False
CONNECTION_STRING = args.mongodb
p_db = args.dbname
p_face_collection = 'embeddings'
p_profile_collection = 'profiles'

p_resume = True if args.resume == 1 else False

p_do_split_dataset = True if args.datasetsplit == 1 else False
p_dataset_ratio = [0.8, 0.2]
p_split_seed = args.datasetseed
orig_path = '../tests/VN-celeb/'
split_path = f'../tests/VN-celeb-split-{p_split_seed}/'
train_path = os.path.join(split_path, 'train')
val_path = os.path.join(split_path, 'val')

# Parameters
p_models = ["VGG-Face", "Facenet", "Facenet512",
            "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
p_model = args.facevectormodel
CNN_OUTPUT_SIZE = 128
CNN_INPUT_SIZE = (160, 160)

p_backends = ['opencv', 'ssd', 'dlib',
              'mtcnn', 'retinaface', 'mediapipe', 'skip']
p_detector_backend = args.facedetectbackend

p_grayscale = False
p_enforce_detection = True if args.enforcefacedetection == 1 else False

p_metrics = ['euclidean', 'cosine']
p_metric = args.similaritymeasure
p_normalize_vectors = True if args.normalizevectors == 1 else False
p_cuda_index = False

p_consensus_test_upper_inclusive = 32

p_export_folder = f'../results/{run_uuid}/'
p_export_json_class2pid = os.path.join(
    p_export_folder, f'train_class2pid.json')
p_export_json_train_faces = os.path.join(p_export_folder, f'train_faces.json')
p_export_json_train_profiles = os.path.join(
    p_export_folder, f'train_profiles.json')
p_export_json_val_faces = os.path.join(p_export_folder, f'val_faces.json')
p_export_json_faiss_kneighbors = os.path.join(
    p_export_folder, f'val_faiss_k32_neighbors.json')
p_export_bin_faiss_index = os.path.join(p_export_folder, f'faiss_index.bin')

p_result_eval = os.path.join(p_export_folder, f'eval.csv')
p_result_accuracy = os.path.join(p_export_folder, f'accuracy.csv')

p_tensorflow_export = os.path.join(
    p_export_folder, f'{p_model}_{start_time.replace(":", "").replace(".", "")}')

if not os.path.exists(p_export_folder):
    os.makedirs(p_export_folder)

print(f"[START] Test run {run_uuid}. Begin at {start_time}")

# %% [markdown]
# ### Dataset Preparation: 80-20 split

# %% [markdown]
# Using `splitfolders` to split dataset into training and validation set with customizable ratio and seed
#
# Here, 80-20 is chosen

# %%

if p_do_split_dataset:
    splitfolders.ratio(
        orig_path, output=split_path, seed=p_split_seed, ratio=(
            p_dataset_ratio[0], p_dataset_ratio[1])
    )
# %%


def get_dataset_stat(base_path):
    name_series = []
    count_series = []

    for root, dirs, files in os.walk(base_path):
        if not dirs:
            person_name = root.split('/')[-1]
            img_count = len(files)

            name_series.append(person_name)
            count_series.append(img_count)

    df = pd.DataFrame({
        'name': name_series,
        'pic_count': count_series
    })

    return df


full_stat = get_dataset_stat(orig_path)
train_stat = get_dataset_stat(train_path)
val_stat = get_dataset_stat(val_path)

full_sum = full_stat['pic_count'].sum()
train_sum = train_stat['pic_count'].sum()
val_sum = val_stat['pic_count'].sum()

print("Full Dataset Description\n", full_stat.describe(), f"\nsum\t{full_sum}")
print("Train Dataset Description\n",
      train_stat.describe(), f"\nsum\t{train_sum}")
print("Validation Dataset Description\n",
      val_stat.describe(), f"\nsum\t{val_sum}")

print(f'{full_sum} {"==" if full_sum == train_sum + val_sum else "!="} {train_sum} + {val_sum}')

if full_sum != train_sum + val_sum:
    input("[ERROR] Dataset count mismatch. Stop process and recreate dataset to avoid wrong results.")


# %% [markdown]
# ### Train

# %% [markdown]
# #### Facial Embedding vectors and Database insertions

# %% [markdown]
# Gather image path and group by class (profile) into a `dict{type, name, images}`.

# %%
base_path = train_path

train_profiles = []

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
        train_profiles.append(profile)

# %% [markdown]
# #### FacialIndex instance (Facenet + FAISS)

# %% [markdown]
# ##### Initialization

# %%

fi = FacialIndex(
    model_name=p_model,
    face_detector_backend=p_detector_backend,
    face_enforce_detection=p_enforce_detection,
    faiss_use_cuda=p_cuda_index,
    faiss_index_type=p_metric,
    faiss_vector_normalize=p_normalize_vectors,
    use_db=p_use_db,
    mongodb_client=CONNECTION_STRING,
    db_name=p_db
)

# %% [markdown]
# ##### Add Profiles and Faces to Index

# %%
if p_resume == True:
    faces_out = fi.get_cached_faces()
    profiles_out = fi.get_cached_profiles()
else:
    faces_out, profiles_out = fi.add_profiles(train_profiles)

# %%
print(
    f'Faces: {sys.getsizeof(faces_out)}B\nProfiles: {sys.getsizeof(profiles_out)}B')

# %%
classname2pid = {}
for profile in list(profiles_out.values()):
    classname2pid[profile['name']] = profile['_id']

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
# #### Testing face query with FAISS index

# %% [markdown]
# ##### Get k-nearest neighbors of every faces in validation dataset for later analysis and optimization

# %%

val_faiss_k_neighbors = {}

for path in tqdm(paths):
    measures, neighbors = fi.query_face_faiss(
        path, k=p_consensus_test_upper_inclusive)

    # matches = face_collection.find(
    #     { "_id": { "$in": [f"face_{idnum}" for idnum in neighbors[0].tolist()] } },
    #     { "profile_id": 1 }
    # )
    # matches = list(matches)

    matches = [faces_out[f'embedding_{idnum}'] for idnum in neighbors.tolist()]

    results = [[np.float64(distance), match['_id'], match['profile_id']]
               for match, distance in zip(matches, measures)]

    spl = path.split('/')
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
    val_truth_pid.append(classname2pid[spl[-2]])

# %%
eval_df = pd.DataFrame(
    {
        "truth_class": val_classnames,
        "truth_pid": val_truth_pid
    },
    index=val_paths2
)

# %%
for i in range(1, p_consensus_test_upper_inclusive + 1):
    eval_df[f'predict_{i}'] = np.nan
    eval_df[f'positive_{i}'] = np.nan

# %%
eval_df.head()

# %% [markdown]
# #### Get Consensus for k=[1,25]

# %%


def get_consensus(result_arr, top_k=[None]):
    distances, embeddings, profiles = list(zip(*result_arr))

    ret = ()
    for k in top_k:
        ret = ret + (mode(profiles[0:k]), )
    return ret


# %%
for key, value in val_faiss_k_neighbors.items():
    predictions = get_consensus(value,  list(
        range(1, p_consensus_test_upper_inclusive + 1)))
    eval_df.at[key, 'predicted'] = True
    for n in range(1, p_consensus_test_upper_inclusive + 1):
        eval_df.at[key, f'predict_{n}'] = predictions[n - 1]

# %%
for n in range(1, p_consensus_test_upper_inclusive + 1):
    eval_df[f'positive_{n}'] = np.where(
        (eval_df['truth_pid'] == eval_df[f'predict_{n}']), True, False)

# %%
eval_df[eval_df['predicted'] == True].head()

# %%
total_count = eval_df[eval_df['predicted'] == True].shape[0]
total_count

# %%
accuracy_series = []
positive_series = []

for n in range(1, p_consensus_test_upper_inclusive + 1):
    k_positive_count = eval_df[eval_df['truth_pid']
                               == eval_df[f'predict_{n}']].shape[0]
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
    index=list(range(1, p_consensus_test_upper_inclusive + 1))
)
result_df

# %%
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
fi.get_model().save(p_tensorflow_export)

# %%

with open(p_export_json_train_faces, 'w') as f:
    json.dump(faces_out, f)

with open(p_export_json_train_profiles, 'w') as f:
    json.dump(profiles_out, f)

print(
    f'Model output written. {len(faces_out)} and {len(profiles_out)} dictionary keys')

# %%
with open(p_export_json_class2pid, 'w') as f:
    json.dump(classname2pid, f)

print("Classname-ProfileId map written to disk")

# %%
# with open(p_export_json_val_faces, 'w') as f:
#     json.dump(val_emb, f)

# print("Validation Face Embeddings: Written {} {}-length vectors".format(len(val_emb), len(embedding)))

# %%
with open(p_export_json_faiss_kneighbors, 'w') as f:
    json.dump(val_faiss_k_neighbors, f)

print(f"Written {len(val_faiss_k_neighbors)} dicts, each with {p_consensus_test_upper_inclusive}x{CNN_OUTPUT_SIZE}-length vectors")

# %%
if p_cuda_index == True:
    index_flat_cpu = faiss.index_gpu_to_cpu(fi.get_index())
    faiss.write_index(index_flat_cpu, p_export_bin_faiss_index)
else:
    faiss.write_index(fi.get_index(), p_export_bin_faiss_index)

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
with open(os.path.join(p_export_folder, 'params.json'), 'w') as f:
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
