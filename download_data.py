from bs4 import BeautifulSoup
import os
import requests
import urllib.request
import cv2
import numpy as np
import base64

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
}

def get_url(query):
    return "https://www.google.co.in/search?q=" + query + "&source=lnms&tbm=isch"


def get_soup(query):
    r =  requests.get(get_url(query), headers=HEADERS)
    r.encoding = 'utf-8'
    return BeautifulSoup(r.text, 'html.parser')


def save_image(url, save_path):
    print("downloading ", url)
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)

    if img is not None:
        cv2.imwrite(save_path, img)
        return True

    return False


def download_images(query, size):
    print("Download images with query: {}".format(query))
    output_directory = "./dataset"
    image_directory = query.replace(" ", "_")
    save_path = os.path.join(output_directory, image_directory)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # using face for better query result
    soup = get_soup(query + " face")
    itags = soup.findAll('img')
    count = 0
    for i, itag in enumerate(itags):
        img_url = itag.get('data-src')
        if img_url:
            # data_type, url = img_url.split("base64,")
            # print(img_url)
            # img_url = data_type + "base64," + base64.b64decode(url)
            img_save_path = os.path.join(save_path, '{}.png'.format(i))
            if save_image(img_url, img_save_path):
                count+=1

    print("Done, downloaded: {}".format(count))


if __name__ == "__main__":
    # Download data from google image search by keyword
    names = [
        "emma watson",
        "christoph waltz",
        "brad pitt",
        "barack obama",
        "donald trump",

        # Asian
        "chipu",
        "son tung mtp",
        "le cong vinh",
        "thao tam",
        "dong nhi",
        "singer min",
        "den vau",
    ]

    names = [
        "madong seok",
        "kim da mi",
        "park seo joon",
        "misthy",
        "minh nghi",
    ]

    size = 20
    for name in names:
        download_images(name, size)
