import os
from owlready2 import *

def load_ontology(path):
    # check for the validity of the given path
    assert os.path.exists(path), (f"'{path}' is not a valid path.")

    # load the ontology
    onto = get_ontology(path).load()

    return onto
