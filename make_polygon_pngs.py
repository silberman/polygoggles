#!/usr/bin/python
"""
Functions for generating random polygon PNGs.

Usage:
draw_random_polygon(max_edges=9, allow_rotation=True, image_width=480, image_height=640)

draw_polygon(num_edges, edge_length=30, rotate_degrees=0,
             image_width=100, image_height=100, show=False)

Default behavior is to draw an approximately-centered red polygon on a white image, output as
a .png to the /images directory.

The number of edges in the polygon is included in the filename.
"""

# imports for python2 backward compatibility
from __future__ import print_function
from __future__ import division

import argparse
import math
import os
from PIL import Image, ImageDraw
import random
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(THIS_DIR, "images")


def draw_random_polygon(max_edges=9, allow_rotation=True, image_width=480, image_height=640):
    num_edges = random.randint(3, max_edges)

    # Figure out some reasonable bounds on the edge_size for an image of the given size.
    max_allowed_edge_length = int(min(image_width, image_height) / 3.3)
    min_allowed_edge_length = min(10, max_allowed_edge_length)
    edge_length = random.randint(min_allowed_edge_length, max_allowed_edge_length)

    if allow_rotation:
        rotate_degrees = random.uniform(0.0, 360.0)
    else:
        rotate_degrees = 0
    return draw_polygon(num_edges, edge_length=edge_length, rotate_degrees=rotate_degrees,
                        image_width=image_width, image_height=image_height)


def draw_polygon(num_edges, edge_length=30, rotate_degrees=0,
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
    # the left-most point and move just onto the image, along with the other points appropriately.
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
    image = Image.new('RGB', (image_width, image_height), 'white')
    drawer = ImageDraw.Draw(image)
    drawer.polygon(vertices, fill=128, outline=128)

    # Write the image to a file.  We include the number of edges as a label in the filename,
    # along with the width and height. The rest of the filename is just to avoid duplicates.
    output_partial_filename = "polygon_%s_%s_%s_%s%s.png" % (
        num_edges, image_width, image_height, int(time.time()), int(random.random() * 100000))
    output_full_filename = os.path.join(IMAGES_DIR, output_partial_filename)
    image.save(output_full_filename, "PNG")
    if show:
        image.show()
    print("Wrote:", output_full_filename)
    return output_full_filename

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('num_images', help='How many images to make.', type=int)
    parser.add_argument('width', help='Pixel width of the images to make.', type=int)
    parser.add_argument('height', help='Pixel height of the imags to make.', type=int)
    args = parser.parse_args()

    for __ in range(args.num_images):
        draw_random_polygon(max_edges=9, image_width=args.width,
                            image_height=args.height, allow_rotation=True)
