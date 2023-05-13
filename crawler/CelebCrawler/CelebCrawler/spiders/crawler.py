import scrapy
from CelebCrawler.items import CelebCrawlerItem


class Celeb(scrapy.Spider):
    name = "celeb"
    path_dt = 'https://nguoinoitieng.tv'
    start_urls = [
        'https://nguoinoitieng.tv/nghe-nghiep/ca-si/son-tung/k8', ]
    index = 1
    LIMIT = 200000

    def parse(self, response):
        celeb = CelebCrawlerItem()
        celeb['_id'] = f'profile_{self.index}'
        celeb['name'] = response.css('div.motangan h2::text').get()
        celeb['job'] = response.css('a.nganhhd::attr("title")').get()
        celeb['address'] = response.css('div.motangan > p::attr("title")').get()
        celeb['dob'] = ' - '.join(response.css('nav.thongtin-right ul li p a[href*=sinh]::text').getall())
        celeb['rank'] = response.xpath('//*[@id="content-left"]/div[3]/div[2]/p[4]/text()').get()
        celeb['fb'] = response.css('a.fbl::attr("href")').get()
        celeb['email'] = response.xpath('//*[@id="content-left"]/div[3]/div[2]/p[6]/text()').get()
        celeb['phone'] = response.xpath('//*[@id="content-left"]/div[3]/div[2]/p[7]/text()').get()
        celeb['image_urls'] = [self.path_dt + relative for relative in response.css('div.about-nnt img::attr("src")').getall()]
        
        if self.index > self.LIMIT:
            return
        
        self.index += 1
        yield celeb

        next_links = self.get_next_links(response)
        if next_links is not None:
            for link in next_links:
                yield scrapy.Request(link, callback=self.parse)

    def get_next_links(self, response):
        links = response.css('div.list-ngaymai div figure figcaption a.tennnt::attr("href")').getall()
        qualified_links = []
        for link in links:
            if self.path_dt + link not in self.start_urls:
                self.start_urls.append(self.path_dt + link)
                qualified_links.append(self.path_dt + link)
        if qualified_links is not None:
            return qualified_links
        return None
