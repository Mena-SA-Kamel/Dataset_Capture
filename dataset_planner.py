import numpy as np
import os
import matplotlib.pyplot as plt

def get_num_objects_per_image():
    num_objects = int(np.random.normal(6, 2, 1))
    if num_objects < 0:
        num_objects = 0
    return num_objects

def initialize_objects():
    if os.path.exists('all_objects.txt'):
        f = open("all_objects.txt", "r")
        all_objects = f.read().splitlines()
        all_objects = np.array(all_objects)
        f.close()
    else:
        office_supplies = ['calculator','pens','binder','eraser','notebooks','stapler','tissues','tissue boxes']
        food_items = ['apple','banana', 'orange','chips','water bottle','water jug','cans']
        clothing = ['running shoes', 'tshirts', 'boots', 'glasses']
        technology_products = ['watch','ipad charger','headphones','airpods','speakers','laptop charger','laptop','keyboard','mouse']
        kitchen_items = ['jars','chocolate bars','gum packs','dish soap','kettle','transparent mug','normal mugs','travel mugs', 'spoons','forks','knives','plates','kitchen towels']
        bathroom_products = ['Tooth paste','Tooth brush','lotion bottles','hand sanitizer','towels', 'toilet paper']
        tools = ['tape','scissors','glue gun','measuring tape', 'clamps', 'screw driver', 'hammer']

        all_objects = []
        all_objects.extend(office_supplies)
        all_objects.extend(food_items)
        all_objects.extend(clothing)
        all_objects.extend(technology_products)
        all_objects.extend(kitchen_items)
        all_objects.extend(bathroom_products)
        all_objects.extend(tools)

        all_objects = np.array(all_objects)
        np.random.shuffle(all_objects)

        np.savetxt('all_objects.txt', all_objects, fmt = "%s")
    return all_objects

# objects = initialize_objects()
# objects_drawn = np.random.choice(objects, get_num_objects_per_image())
#
#



