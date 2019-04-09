import requests
import re
from bs4 import BeautifulSoup
import codecs
import glob
import os
# print('Beginning file download with requests')

# url = 'http://dmccooey.com/polyhedra/index.html'  
# r = requests.get(url)

# with open('./dmccooey/polyhedra/index.html', 'wb') as f:  
#     f.write(r.content)

# # Retrieve HTTP meta-data
# print(r.status_code)  
# print(r.headers['content-type'])  
# print(r.encoding)  



""" This grabbed all the polygonal groups as per dmccooey
  base_url = "http://dmccooey.com/polyhedra/"
  base_path = "./dmccooey/polyhedra/"

  f=codecs.open("./dmccooey/polyhedra/index.html", 'r', 'utf-8')
  document= BeautifulSoup(f.read(), 'html.parser')
  pattern = re.compile("\w*\.html")

  for e, link in enumerate(document.findAll('a')):
    if pattern.fullmatch(link["href"]):
      file_name = link["href"].split(".")[0] + ".txt"
      txt_url = base_url  + file_name
      r = requests.get(txt_url)
      if not r.status_code == 200:
        r = requests.get(base_url + link["href"])
        print(r.status_code)
        if r.status_code == 200:
          with open(base_path + link["href"], 'wb') as f:  
              f.write(r.content)
      # if text_url and not local url dowload it.
"""
base_url = "http://dmccooey.com/polyhedra/"
base_path = "./dmccooey/polyhedra/"
configfiles = glob.glob('./dmccooey/polyhedra/*.html')

for c in configfiles:
  directory = base_path + c.split("/")[-1].split(".")[0]
  f=codecs.open(c, 'r', 'utf-8')
  document= BeautifulSoup(f.read(), 'html.parser')
  pattern = re.compile("\w*\.html")

  if not os.path.exists(directory):
      os.makedirs(directory)

  for e, link in enumerate(document.findAll('a')):
    if pattern.fullmatch(link["href"]):
      print(link["href"], link.text)
      file_name = link["href"].split(".")[0] + ".txt"
      txt_url = base_url  + file_name
      r = requests.get(txt_url)
      path_name = directory +"/" + file_name
      if r.status_code == 200:
        with open(path_name, 'wb') as f:  
            f.write(r.content)


