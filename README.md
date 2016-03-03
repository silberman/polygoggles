# polygoggles
CNN for detecting types of polygons.

Currently just a script to generate images of polygons for training.

## Installation

Ideally from a python3 virtualenv:

```
pip3 install -r requirements.txt
pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp35-none-any.whl
```

## Example Usage

From a command line, run make_polygon_pngs.py with the number of
images to make:
```
(polygoggles) ~/polygoggles$ python make_polygon_pngs.py 20
```
