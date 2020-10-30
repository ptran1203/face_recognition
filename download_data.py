from bs4 import BeautifulSoup
import os
import requests
import urllib.request
import cv2
import numpy as np
import json
import threading


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
}
SAVE_DIR = "./dataset"


def get_url(query):
    return "http://www.bing.com/images/search?q=" + query + "&FORM=HDRSC2"


def get_soup(query):
    r = requests.get(get_url(query), headers=HEADERS)
    r.encoding = "utf-8"
    return BeautifulSoup(r.text, "html.parser")


def save_image(url, save_path):
    # print("downloading ", url)
    try:
        req = urllib.request.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
    except Exception as e:
        print("Could not get image from {}, ERROR: {}".format(url, str(e)))
        img = None

    if img is not None:
        cv2.imwrite(save_path, scale(img))
        return True

    return False


def scale(img):
    w, h = img.shape[:2]
    if w > h:
        return image_resize(img, height=512)

    return image_resize(img, width=512)


def download_images(query):
    print("Download images with query: {}".format(query))

    image_directory = query.replace(" ", "_")
    save_path = os.path.join(SAVE_DIR, image_directory)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # using face for better query result
    soup = get_soup(query)
    image_tags = soup.findAll("a", {"class": "iusc"})
    count = 0
    for i, itag in enumerate(image_tags):
        m = json.loads(itag["m"])
        # mobile image
        murl = m["murl"]
        if murl:
            img_save_path = os.path.join(save_path, "{}_{}.png".format(image_directory,i))
            if save_image(murl, img_save_path):
                count += 1

    print("Done, downloaded: {}".format(count))


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def download_images_in_list(names):
    for name in names:
        if not os.path.isdir(os.path.join(SAVE_DIR, name.replace(" ", "_"))):
            name and download_images(name)
        else:
            print("Data already exist")


if __name__ == "__main__":
    # Download data from google image search by keyword
    with open("identity_list.txt", "r") as f:
        names = f.read().split("\n")

    list_size = len(names)
    haft = list_size // 2

    part1 = names[:haft]
    part2 = names[haft:]

    t1 = threading.Thread(target=download_images_in_list, args=(part1,), kwargs={})
    t2 = threading.Thread(target=download_images_in_list, args=(part2,), kwargs={})

    print("Total identities: {}".format(list_size))

    t1.start()
    t2.start()
    t1.join()
    t2.join()
