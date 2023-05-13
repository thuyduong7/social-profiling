from pymongo import MongoClient, ReturnDocument
import pymongo
import gridfs
from datetime import datetime
import faiss
import numpy as np
# from deepface.basemodels import Facenet
from deepface import DeepFace
from deepface.commons import functions
from tqdm import tqdm
from copy import deepcopy
import os
from .error import FacialIndexError, ProfileFoundNoFaceError, ImagePreprocessFaceError


class FacialIndex():
    CONFIG_COLLECTION_NAME = 'config'
    PROFILE_COLLECTION_NAME = 'profiles'
    FACIAL_EMBEDDINGS_COLLECTION_NAME = 'embeddings'
    CNN_IO_SIZE_MAP = {
        # [ inputX, inputY, output ]
        'VGG-Face': [224, 224, 2622],
        'Facenet': [160, 160, 128],
        'OpenFace': [96, 96, 128],
        'DeepFace': [152, 152, 4096],
        'DeepID': [55, 47, 160],
        'Dlib': [150, 150, 128]
    }

    # models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
    # detector_backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe', 'skip']

    # instance variables
    # db
    # fs
    # config_collection
    # profile_collection
    # face_collection
    # model_name: string name of model
    # cnn_input_size: input size of model
    # cnn_output_size: output size of model
    # face_detector_backend: string name for face detector backend, in face preprocess step
    # face_enforce_detection: T/F enforce face detection in face preprocess step
    # faiss_use_cuda: T/F use GPU acceleration for faiss index
    # faiss_vector_normalize: faiss.normalize_L2() the embeddings before index insert/query
    # faiss_gpu_res: faiss.StandardGpuResource()
    # index
    # model

    def __init__(self,
                 model_name='Facenet',
                 face_detector_backend='skip',
                 face_enforce_detection=True,
                 #  face_retry_on_not_found=False,
                 faiss_use_cuda=False,
                 faiss_index_type='cosine',
                 faiss_vector_normalize=True,
                 use_db=False,
                 mongodb_client=False,
                 db_name=False):
        self.profile_counter = 0
        self.face_counter = 0
        self.faces_map = {}
        self.profiles_map = {}

        self.use_db = use_db
        if use_db == True:
            # Accepts either a mongodb server connection string
            # or a MongoClient
            if isinstance(mongodb_client, MongoClient):
                self.client = mongodb_client
            elif isinstance(mongodb_client, str):
                self.client = MongoClient(mongodb_client)

            # db connection to the database of interest
            self.db = self.client[db_name]

            # GridFS instance to the same database
            self.fs = gridfs.GridFS(self.db)

            # create collection instances for config, profile, and facial embeddings
            self.config_collection = self.db[self.CONFIG_COLLECTION_NAME]
            self.profile_collection = self.db[self.PROFILE_COLLECTION_NAME]
            self.face_collection = self.db[self.FACIAL_EMBEDDINGS_COLLECTION_NAME]

            # get or create new config document
            config = self.config_collection.find_one_and_update(
                filter={'_id': 'config'},
                update={"$set": {'lastAccessed': datetime.now()}},
                upsert=True,
                return_document=ReturnDocument.AFTER
            )

            counter_doc = self.config_collection.find_one(
                filter={'_id': 'config_counter'},
            )
            if counter_doc is None:
                # initialize profile and face counter to 0
                self.config_collection.insert_one(
                    {
                        '_id': 'config_counter',
                        'profile_counter': self.profile_counter,
                        'face_counter': self.face_counter
                    }
                )
            else:
                self.profile_counter = counter_doc['profile_counter']
                self.face_counter = counter_doc['face_counter']

        self.model_name = model_name
        model_io_size = self.CNN_IO_SIZE_MAP[model_name]
        self.cnn_input_size = (model_io_size[0], model_io_size[1])
        self.cnn_output_size = model_io_size[2]
        self.face_detector_backend = face_detector_backend
        self.face_enforce_detection = face_enforce_detection
        # self.face_retry_on_not_found = face_retry_on_not_found
        self.faiss_vector_normalize = faiss_vector_normalize
        self.faiss_use_cuda = faiss_use_cuda

        # Initialize and construct faiss index
        if faiss_index_type == 'euclidean':
            self.index = faiss.IndexFlatL2(self.cnn_output_size)
        elif faiss_index_type == 'cosine':
            self.index = faiss.IndexFlatIP(self.cnn_output_size)
        else:
            self.index = faiss.IndexFlatL2(self.cnn_output_size)

        self.index = faiss.IndexIDMap(self.index)
        if faiss_use_cuda == True:
            self.faiss_gpu_res = faiss.StandardGpuResource()
            self.index = faiss.index_cpu_to_gpu(
                self.faiss_gpu_res, 0, self.index)
        if use_db == True:
            self.__construct_index_from_db()

        # Init Facenet model
        self.model = DeepFace.build_model(model_name)

        print(
            "FacialIndex instance created with the following parameters and variables: \n",
            f"model_name={model_name}\n",
            f"face_detector_backend={face_detector_backend}\n",
            f"face_enforce_detection={face_enforce_detection}\n",
            f"faiss_use_cuda={faiss_use_cuda}\n",
            f"faiss_index_type={faiss_index_type}\n",
            f"faiss_vector_normalize={faiss_vector_normalize}\n",
            f"use_db={use_db}", f" at {mongodb_client}, dbname={db_name}\n" if use_db == True else "\n",
            f"profile_counter={self.profile_counter}, face_counter={self.face_counter}",
            f"index size: {self.get_index().ntotal}"
        )

    def __del__(self):
        if self.use_db == True:
            del self.client
            del self.db
            del self.fs
            del self.config_collection
            del self.profile_collection
            del self.face_collection

        del self.index
        if self.faiss_use_cuda == True:
            del self.faiss_gpu_res

        del self.model

        print(
            "FacialIndex instance destroyed. RAM freed, VRAM claimed by TensorFlow may not be freed (try exit() python instance)"
        )

    def __construct_index_from_db(self):
        # Fetch all embeddings from database
        femb_count = self.face_collection.count_documents(
            filter={'type': 'embedding'},
        )
        if femb_count <= 0:
            return
        faces = self.face_collection.find()
        faces = list(faces)

        self.__add_faces_to_faiss(faces)
        print(f"[INFO] Constructed FAISS index from database with {len(faces)} faces. Current index size {self.get_index().ntotal}")

        profiles = self.profile_collection.find()
        profiles = list(profiles)

        t_faces_map = {}
        t_profiles_map = {}

        for face in faces:
            t_faces_map[ face['_id'] ] = face

        for profile in profiles:
            t_profiles_map[ profile['_id'] ] = profile
        
        self.faces_map = {**self.faces_map, **t_faces_map}
        self.profiles_map = {**self.profiles_map, **t_profiles_map}
        
        print(f'[INFO] Cached {len(t_profiles_map)} profiles with {len(t_faces_map)} faces to memory')

        return

    # Return 2 numpy array/vector: target, region
    def __get_face_embedding(self, img, grayscale=False,
                             enforce_detection=True, detector_backend='opencv',
                             return_region=False, align=True):
        try:
            target, region = functions.preprocess_face(
                img=img, target_size=self.cnn_input_size, grayscale=grayscale,
                enforce_detection=enforce_detection,
                detector_backend=detector_backend, return_region=True, align=align)
        except ValueError:
            raise ImagePreprocessFaceError(img)
        target = self.model.predict(target)[0, :]

        if return_region == True:
            return target, region
        else:
            return target

    # Add face embeddings to the faiss index, perform vector normalization if needed
    def __add_faces_to_faiss(self, faces_dict_list):
        t_ids = []
        t_embeddings = []

        for face in faces_dict_list:
            fid = face['_id']
            # id is in format embedding_xxxxx where xxxxx is number
            fid = fid.split('_')[-1]
            embedding = face['values']

            t_ids.append(fid)
            t_embeddings.append(embedding)

        t_ids = np.array(t_ids, dtype='int64')
        t_embeddings = np.array(t_embeddings, dtype='f')

        if self.faiss_vector_normalize == True:
            faiss.normalize_L2(t_embeddings)

        self.index.add_with_ids(t_embeddings, t_ids)

    # Process Profile:
    # Add fields and convert image paths to face embeddings ready for db
    # subject is a Dict with various key-value pairs of informations
    # regarding the subject, all will be inserted to db
    #
    # subject must include a key-value pair where
    # key = 'images'
    # values = path to the images of the subject
    # note that path must be valid from the working directory
    # {
    #   [...]
    #   'images': [ /path/to/image/1, /path/to/image/2, ... ]
    # }
    # Returns a profile with a list of facial embeddings attached to it
    def __process_profile(self, subject, keep_id_field=True):
        # Deep copy of subject
        profile = deepcopy(subject)

        # Pop the images list consisting of paths to the images
        images = profile.pop('images', None)
        # Add type 'profile'
        profile['type'] = 'profile'
        # use existing id if already present in dict AND keep_id_field == True
        if keep_id_field == True and '_id' in profile:
            pid = profile['_id']
        else:
            # profile.pop('_id')
            # Add new id to profile, incr counter aswell
            pid = f'profile_{self.profile_counter}'
            profile['_id'] = pid
            self.profile_counter += 1

        # # Use existing id if already present in dict
        # if '_id' not in profile:
        #     # profile.pop('_id')
        #     # Add new id to profile, incr counter aswell
        #     pid = f'profile_{self.profile_counter}'
        #     profile['_id'] = pid
        #     self.profile_counter += 1
        # else:
        #     pid = profile['_id']

        # Empty array storing facial embeddings document,
        # each embedding is a document, with a list w/ len=CNN_OUTPUT_SIZE stored in 'values'
        facial_embeddings = []

        # Count of dropped images from this profile
        dropped_images = 0

        # Drop image on face not found
        # For each image path
        for image in images:
            # Process then get embedding for the face
            try:
                # Try getting embedding using instance settings
                embedding, region = self.__get_face_embedding(
                    img=image, grayscale=False, return_region=True,
                    detector_backend=self.face_detector_backend, enforce_detection=self.face_enforce_detection)
            except ImagePreprocessFaceError:
                # If exception occurs set detector backend to 'mtcnn'. Slower but more accurate
                dropped_images += 1
                continue

            # append to array
            eid = f'embedding_{self.face_counter}'
            facial_embeddings.append(
                {
                    '_id': eid,
                    'type': 'embedding',
                    'profile_id': pid,
                    'image_path': image,
                    # tolist() converts np.array to native Python list
                    'values': embedding.tolist(),
                    # converts every single value from numpy datatypes to python native
                    'region': [np.asscalar(i) if isinstance(i, np.generic) else i for i in region]
                }
            )
            self.face_counter += 1

        # If no image is found, raise Exception
        if not facial_embeddings:
            raise ProfileFoundNoFaceError(profile['name'])

        # profile['_facial_embeddings'] = facial_embeddings

        return profile, facial_embeddings, dropped_images

    # Returns faces and profiles
    # Note that returned faces has not been normalized
    # Although it is normalized before insertion to index if needed
    def add_profiles(self, subjects_with_image_paths, keep_id_field=True, retry_img_on_face_not_found=False):
        # Temporary map for faces and profiles added currently being added
        t_faces_map = {}
        t_profiles_map = {}
        profile_found_no_face_count = 0
        images_dropped_count = 0

        # iterate through the subject
        for subject in tqdm(subjects_with_image_paths, desc="Adding Subjects and Faces to index"):
            # do process profile, returning the profile and the embeddings
            try:
                profile, embeddings, dropped_images = self.__process_profile(
                    subject, keep_id_field=keep_id_field)
                images_dropped_count += dropped_images
            except ProfileFoundNoFaceError:
                profile_found_no_face_count += 1
                images_dropped_count += len(subject['images'])
                continue

            # extract id field from profile, for addition to map
            pid = profile['_id']
            t_profiles_map[pid] = profile
            # iter through embeddings
            for emb in embeddings:
                # extract id field from embedding, for addition to map
                eid = emb['_id']
                t_faces_map[eid] = emb

        print(
            f"[INFO] {len(t_profiles_map)} profiles with {len(t_faces_map)} faces processed.")
        print(
            f"[INFO] {profile_found_no_face_count} profiles dropped due to having found no faces in any of the images.")
        print(
            f"[INFO] {images_dropped_count} images dropped in total due to having found no faces.")

        self.faces_map = {**self.faces_map, **t_faces_map}
        self.profiles_map = {**self.profiles_map, **t_profiles_map}

        self.__add_faces_to_faiss(list(t_faces_map.values()))
        print(f"[INFO] {len(t_profiles_map)} profiles with {len(t_faces_map)} faces inserted to index. Current index size {self.index.ntotal}")

        if self.use_db == True:
            dbq_profile = list(t_profiles_map.values())
            dbq_face = list(t_faces_map.values())
            try:
                self.profile_collection.insert_many(documents=dbq_profile)
                self.face_collection.insert_many(documents=dbq_face)
                self.config_collection.update_one(
                    { '_id': 'config_counter' }, 
                    { '$set': {
                        'profile_counter': self.profile_counter,
                        'face_counter': self.face_counter
                    } },
                    upsert=True)
            except pymongo.errors.ServerSelectionTimeoutError:
                print(
                    "[ERROR] Unable to connect to database, insertion retry possible")
                pass
            except pymongo.errors.InvalidDocument as e:
                print(
                    f"[ERROR] InvalidDocument {e}"
                )
                pass
            except:
                print(f"[ERROR] Error occurred while inserting to database")
                pass
            else:
                print(
                    f"[SUCCESS] Inserted {len(dbq_profile)} profiles and {len(dbq_face)} faces to db")

        return t_faces_map, t_profiles_map

    def query_face_faiss(self, img, k=1):
        target = self.__get_face_embedding(
            img, enforce_detection=self.face_enforce_detection, detector_backend=self.face_detector_backend)

        target = np.array(target, dtype='f')
        target = np.expand_dims(target, axis=0)

        if self.faiss_vector_normalize == True:
            faiss.normalize_L2(target)

        measures, neighbors = self.index.search(target, k)

        return measures[0], neighbors[0]

    # GETTERS

    def get_model(self):
        return self.model

    def get_index(self):
        return self.index
    
    def get_cached_faces(self):
        return deepcopy(self.faces_map)
    
    def get_cached_profiles(self):
        return deepcopy(self.profiles_map)

    # # @param subject_with_image_paths Subject dict with 'images' list containing paths to the images
    # #
    # def add_profile(self, subject_with_image_paths):
    #     profiles_map = {}
    #     faces_map = {}

    #     # 1. Process profile by converting image path ('images') array to '_facial_embeddings' array
    #     processed_profile, processed_embeddings = self.__process_profile(subject_with_image_paths)

    #     pid = f'profile_{self.profile_counter}'
    #     processed_profile['_id'] = pid
    #     # self.profile_counter += 1

    #     for emb in processed_embeddings:
    #         eid = f'embedding_{self.face_counter}'
    #         emb['_id'] = eid
    #         self.face_counter += 1
    #         emb['profile_id'] = pid

    #         # img_path = emb['image_path']
    #         # img_fsid = self.fs.put(open(img_path, 'r'))
    #         # emb['image_fsid'] = img_fsid

    #         faces_map[eid] = emb

    #     if self.use_db == True:
    #         dbq_profile = processed_profile

    #     with self.client.start_session() as session:
    #         with session.start_transaction():
    #             self.profile_collection.insert_one(
    #                 processed_profile,
    #                 session=session
    #             )

    #     pass
