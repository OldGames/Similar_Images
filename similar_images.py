# Copyright 2020

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
# and associated documentation files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial 
# portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from PIL import Image
from pathlib import Path
from scipy import spatial
from pprint import pprint
from collections import defaultdict

import os
import textwrap
import argparse
import imagehash
import numpy as np

IMAGE_EXTENSIONS = [".jpeg", ".jpg", ".gif", ".png", ".bmp"]

DEFAULT_SENSITIVITY = 4
DEFAULT_HTML_REPORT_FILE = "report.html"

BITS_PER_BYTE = 8

def _bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

class HashedImage:
    _hash_size = 8

    def __init__(self, path, hash_func = None):
        self._path = path
        self._hash_func = hash_func if hash_func is not None else imagehash.phash 
        self._hash = self._hash_func(Image.open(path), hash_size = self._hash_size)
        self._hash_as_bit_arr = _bin_array(int(str(self._hash), base = 16), self._hash_size * BITS_PER_BYTE)

    @property
    def path(self):
        return self._path

    @property
    def hash(self):
        return self._hash

    @property
    def hash_func(self):
        return self._hash_func.__name__

    # Needed in order to convert this object into a numpy array for the use of KDTree
    def __getitem__(self, index):
        return self._hash_as_bit_arr[index]

    # Needed in order to convert this object into a numpy array for the use of KDTree
    def __len__(self):        
        return len(self._hash_as_bit_arr)
    
    def __repr__(self):
        return "HashedImage({}, {} = {})".format(self.path, self.hash_func, self._hash)

    def __eq__(self, other):
        if self._hash_func != other._hash_func:
            return False
        return os.path.realpath(self.path) == os.path.realpath(other.path)

    def __hash__(self):
        return hash((self.path, self._hash_func))

def _list_images(path):
    suffix_list = set(IMAGE_EXTENSIONS)
    p = Path(path)
    for subp in p.rglob('*'):
        ext = subp.suffix
        if ext.lower() in suffix_list:
            yield str(subp)

def get_similar_images(path_to_existing_images, path_to_new_images = None, sensitivity = DEFAULT_SENSITIVITY):
    images = {}
    for img_path in _list_images(path_to_existing_images):
        images[img_path] = HashedImage(img_path)

    if path_to_new_images is not None:
        for img_path in _list_images(path_to_new_images):
            images[img_path] = HashedImage(img_path)
    
    image_list = list(images.values())
    tree = spatial.KDTree(image_list)

    res = defaultdict(list)
    for img in _list_images(path_to_new_images if path_to_new_images is not None else path_to_existing_images):
        similar_img_ids = tree.query_ball_point(images[img], r = sensitivity)
        for id in similar_img_ids:
            if images[img] != image_list[id]:
                res[img].append(image_list[id])
    return res

def similar_images_pairs(similar_img_map):
    similar_pairs = set()
    for base_image, similar_to_list in similar_img_map.items():
        for similar_img in similar_to_list:
            similar_pairs.add(frozenset((base_image, similar_img.path)))
    return similar_pairs

def output_as_html(output_html_path, similar_pairs):
    with open(output_html_path, "w") as f:
        f.write("<!doctype html>\n")
        f.write("<html>\n")
        f.write("\t<head>\n")
        f.write("\t\t<title>Similar Images</title>\n")
        f.write(textwrap.indent(textwrap.dedent("""
                    <style>
                        body {
                            font-family: Arial
                        }
                        table, td {
                            border: 1px solid black;
                            text-align: center;
                            padding: 10px;
                        }
                    </style>\n"""), "\t\t"))
        f.write("\t</head>\n")
        f.write("\t<body>\n")

        f.write("\t\t<h1>Similar Images Report</h1>\n".format())
        f.write("\t\t<table width='100%'>\n")
        
        for pair in similar_pairs:
            f.write("\t\t\t<tr>\n")

            for elem in pair:
                f.write("\t\t\t\t<td>\n")
                f.write("\t\t\t\t\t<img src='file://{}' height='400' />\n".format(os.path.realpath(elem)))
                f.write("\t\t\t\t\t<br />\n")
                f.write("\t\t\t\t\t{}\n".format(elem))
                f.write("\t\t\t\t</td>\n")

            f.write("\t\t\t</tr>\n")

        f.write("\t\t</table>\n")

        f.write("\t</body>\n")
        f.write("</html>\n")
    
    print ("HTML report saved to: {}".format(output_html_path))

def main(path_to_existing_images, path_to_new_images, sensitivity, output_html_path):
    similar_img_map = get_similar_images(path_to_existing_images, path_to_new_images, sensitivity)
    similar_pairs = similar_images_pairs(similar_img_map)

    if len(similar_pairs) > 0:
        print("The following similar pairs were found:")
        for pair in similar_pairs:
            print("(*) {} ==~ {}".format(*pair))

        if output_html_path:
            output_as_html(output_html_path, similar_pairs)
    else:
        print("No similar pairs were found")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Search for duplicate images ({}) in a given path'.format(", ".join(IMAGE_EXTENSIONS)))
    parser.add_argument('-p', '--path', help='Path to folder which contains images to be tested for similarity', required = True)
    parser.add_argument('-t', '--test_path', help='If provided, path to folder which contains images to be tested for similarity against the PATH images.'
                                                   'If excluded, PATH images will be tested for similarity against themselves.')
    parser.add_argument('-s', '--sensitivity', help='Sensitivity for similarity test (lower is more sensitive, default: {})'.format(DEFAULT_SENSITIVITY), 
                                type = int, choices = range(1, 10), default = DEFAULT_SENSITIVITY)
    parser.add_argument('--html', help = 'Output an HTML report to the given HTML path (default: {})'.format(DEFAULT_HTML_REPORT_FILE), 
                                action = 'store', nargs = '?', const = DEFAULT_HTML_REPORT_FILE, default = None)
    args = parser.parse_args()
    main(args.path, args.test_path, args.sensitivity, args.html)
