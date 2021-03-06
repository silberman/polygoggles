"""
Functions for generating random polygon PNGs.

Usage:
make_many_random_polygons(num_to_make, directory, image_width, image_height, allow_rotation)

The number of edges in the polygon is included in the filename.
"""

import argparse
import math
import os
from PIL import Image, ImageDraw
import random
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_IMAGES_DIR = os.path.join(THIS_DIR, "images")

PIL_IMAGE_MODE = "L" # RGB

def make_many_random_polygons(num_to_make, directory, image_width, image_height, allow_rotation,
                              min_edges=3, max_edges=9):
    polygon_filenames_made = []
    batch_id = "%sz%s" % (int(time.time()), int(random.random() * 10000000))
    for polygon_making in range(num_to_make):
        # Pick a random number of edges, constant edge length, and rotation
        num_edges = random.randint(min_edges, max_edges)
        # Figure out some reasonable bounds on the edge_size for an image of the given size.
        max_allowed_edge_length = int(min(image_width, image_height) / 3.3)
        min_allowed_edge_length = min(10, max_allowed_edge_length)
        edge_length = random.randint(min_allowed_edge_length, max_allowed_edge_length)

        if allow_rotation:
            rotate_degrees = random.uniform(0.0, 360.0)
        else:
            rotate_degrees = 0

        # Make the filename.  We include the number of edges as a label in the filename,
        # along with the width and height. The rest of the filename is just to avoid duplicates.
        output_partial_filename = "polygon_%s_%s_%s_%s_%s.png" % (num_edges, image_width,
                    image_height, polygon_making, batch_id)
        output_full_filename = os.path.join(directory, output_partial_filename)
        filename_made = draw_polygon(num_edges,
                                     output_full_filename,
                                     edge_length=edge_length,
                                     rotate_degrees=rotate_degrees,
                                     image_width=image_width,
                                     image_height=image_height)

def draw_polygon(num_edges, to_filename, edge_length=30, rotate_degrees=0,
                 image_width=100, image_height=100, show=False):
    """
    Draw a polygon and save it as a PNG.  Return the filename created.
    """
    assert num_edges > 2

    # first we make an unrotated polygon with the first point on the origin.
    x, y = 0, 0
    vertices = [] # list of tuples
    for angle in _angles_of_a_polygon(num_edges):
        x += math.cos(math.radians(angle)) * edge_length
        y += math.sin(math.radians(angle)) * edge_length
        vertex = (x, y)
        vertices.append(vertex)

    # rotate it
    vertices = _rotate_polygon(vertices, rotate_degrees)

    # offset it such that it's near the center. Having rotated around the origin (top left in PIL),
    # it may currently have a negative x or y, so "off" the visible image canvas.  So we'll find
    # the left-most point and move it to a bit left of center, and the other points appropriately.
    center_x = image_width / 2
    center_y = image_height / 2
    lowest_x_seen, y_of_lowest_x = None, None
    for x, y in vertices:
        if lowest_x_seen is None or x < lowest_x_seen:
            lowest_x_seen = x
            y_of_lowest_x = y
    # calculate the offsets we need and apply to each vertex to approximately center the polygon.
    x_offset = center_x - lowest_x_seen - edge_length
    y_offset = center_y - y_of_lowest_x
    vertices = [(x + x_offset, y + y_offset) for x, y in vertices]

    # draw it
    image = Image.new(PIL_IMAGE_MODE, (image_width, image_height), 'white')
    drawer = ImageDraw.Draw(image)
    drawer.polygon(vertices, fill=128, outline=128)
    image.save(to_filename, "PNG")
    if show:
        image.show()

    return to_filename

def _angles_of_a_polygon(num_edges):
    """
    Return a tuple of float angles (in degrees) for a polygon of the given number of edges.

    For example, a hexagon has angles (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
    """
    assert num_edges > 2
    # first see if we have this answer cached already
    if num_edges in _angles_of_a_polygon.cache:
        return _angles_of_a_polygon.cache[num_edges]
    step = 360. / num_edges
    angles_list = [0]
    next_angle = step
    while next_angle < 360:
        angles_list.append(next_angle)
        next_angle += step
    # turn the list of angles to a tuple for immutability, since we'll be caching it and re-using
    angles = tuple(angles_list)

    # add to cache and return
    _angles_of_a_polygon.cache[num_edges] = angles
    return angles
_angles_of_a_polygon.cache = {} # num_edges: (tuple of angles)

def _rotate_polygon(vertices, degrees):
    """
    Rotates the given polygon which consists of corners represented as (x,y) vertices,
    around the origin, clock-wise, theta degrees.

    Returns a new list of vertices.
    """
    theta = math.radians(degrees)
    rotated_vertices = []
    for x, y in vertices:
        rotated_vertex = (x * math.cos(theta) - y * math.sin(theta), x * math.sin(theta) + y * math.cos(theta))
        rotated_vertices.append(rotated_vertex)
    return rotated_vertices

def make_collection(image_width, image_height, num_train_images, num_test_images,
                    root_dir=BASE_IMAGES_DIR, allow_rotation=True, min_edges=3, max_edges=9,
                    training_num_edges_limited_to_range=None):
    """
    A collection is a directory of directories of images, with all images of the same shape.

    For instance, making a collection of 28x28 images will create a directory like:
    root_dir/coll_28_28_1457049237/train/1000_train_images
    root_dir/coll_28_28_1457049237/test/100_test_images

    Return the path to the collection directory.

    If a tuple of training_num_edges_limited_to_range like (min_training_edges, max_training_edges)
    is given, then the training polygons will be limited to that range.
    """
    assert num_train_images > 0 and num_test_images > 0
    collection_dir_name = "coll_%s_%s_%s" % (image_width, image_height, int(time.time()))
    full_collection_dir = os.path.join(root_dir, collection_dir_name)
    train_dir = os.path.join(full_collection_dir, "train")
    test_dir = os.path.join(full_collection_dir, "test")

    # Note: will have collection_dir_name duplicates if you're making more than 1 collection/sec
    assert not os.path.exists(full_collection_dir)
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    if training_num_edges_limited_to_range is not None:
        min_training_edges, max_training_edges = training_num_edges_limited_to_range
    else:
        min_training_edges, max_training_edges = min_edges, max_edges

    print("Making %s training images..." % num_train_images)
    train_names = make_many_random_polygons(num_train_images, train_dir, image_width, image_height,
                                            allow_rotation, min_training_edges, max_training_edges)
    print("Making %s testing images..." % num_test_images)
    test_names = make_many_random_polygons(num_test_images, test_dir, image_width, image_height,
                                           allow_rotation, min_edges, max_edges)
    print("Wrote collection to:", full_collection_dir)
    return full_collection_dir

def main():
    """
    Make a collection of images based on the command line args.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('num_images', help='How many images to make.', type=int)
    parser.add_argument('width', help='Pixel width of the images to make.', type=int)
    parser.add_argument('height', help='Pixel height of the imags to make.', type=int)
    parser.add_argument('--norotate', help='Do not rotate polygons.', action='store_true')
    args = parser.parse_args()
    assert args.num_images > 1

    # We'll put ~20% of these images in a /test subdirectory, and ~80% in /train
    num_test_images = math.ceil(0.2 * args.num_images)
    num_train_images = args.num_images - num_test_images

    allow_rotation = not args.norotate
    collection_base = make_collection(args.width, args.height, num_train_images, num_test_images,
                                      allow_rotation=allow_rotation)


if __name__ == "__main__":
    main()
