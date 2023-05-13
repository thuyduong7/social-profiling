# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class CelebCrawlerItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    _id = scrapy.Field()
    name = scrapy.Field()
    job = scrapy.Field()
    address = scrapy.Field()
    dob = scrapy.Field()
    rank = scrapy.Field()
    fb = scrapy.Field()
    email = scrapy.Field()
    phone = scrapy.Field()

    image_urls = scrapy.Field()
    images = scrapy.Field()
