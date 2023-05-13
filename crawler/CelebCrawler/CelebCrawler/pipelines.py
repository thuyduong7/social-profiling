# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import scrapy
from itemadapter import ItemAdapter
from scrapy.pipelines.images import ImagesPipeline
from scrapy.exceptions import DropItem
import json
import os
from urllib.parse import urlparse
import hashlib
import logging
import pymongo
from slugify import slugify


class MongoPipeline:
    collection_name = 'profiles_redo'

    def __init__(self, mongo_uri, mongo_db):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db

    @classmethod
    def from_crawler(cls, crawler):
        # pull in information from settings.py
        return cls(
            mongo_uri=crawler.settings.get('MONGO_URI'),
            mongo_db=crawler.settings.get('MONGO_DATABASE')
        )

    def open_spider(self, spider):
        # initializing spider
        # opening db connection
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]

    def close_spider(self, spider):
        # clean up when spider is closed
        self.client.close()

    def process_item(self, item, spider):
        # how to handle each post
        # self.db[self.collection_name].insert_one(dict(item))
        item_as_dict = dict(item)
        item_id = item_as_dict['_id']

        self.db[self.collection_name].replace_one(
            filter={'_id': item_id},
            replacement=item_as_dict,
            upsert=True
        )

        logging.debug(f'Post added/updated to MongoDB')
        return item

class MyImagesPipeline(ImagesPipeline):
    def get_media_requests(self, item, info):
        for image_url in item['image_urls']:
            yield scrapy.Request(image_url)
    
    def item_completed(self, results, item, info):
        image_paths = [x['path'] for ok, x in results if ok]
        if not image_paths:
            raise DropItem("Item contains no images")
        item['images'] = image_paths
        return item
    
    def file_path(self, request, response=None, info=None, *, item=None):
        # path = urlparse(request.url).path
        # not_ext, ext = os.path.splitext(path)
        # tokens = not_ext.split('/')
        # filename = tokens[-1] + '-' + tokens[-2] + ext
        
        face_owner_name = slugify(item['name'], separator='_')
        face_owner_job = slugify(item['job'], separator='_')
        face_owner_dob = slugify(item['dob'], separator='_')
        face_owner_rank = slugify(item['rank'], separator='_')

        basename = os.path.basename(urlparse(request.url).path)
        name, ext = os.path.splitext(basename)
        hash_str = hashlib.sha1(request.url.encode('utf-8')).hexdigest()
        
        directory_name = f'{face_owner_job}-{face_owner_name}-{face_owner_dob}-{face_owner_rank}'
        filename = f'{face_owner_name}_{hash_str}{ext}'
        
        return f'full/{directory_name}/{filename}'
        # return 'files/' + os.path.basename(urlparse(request.url).path)