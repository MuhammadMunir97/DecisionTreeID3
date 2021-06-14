from typing import ClassVar
import math
import copy
import sys

class Node:
    def __init__(self, class_values, targeted_rows, remaining_attributes, total_classes):
        self.attribute_value = None
        self.childs = None
        self.targeted_rows = targeted_rows
        self.total_classes = total_classes
        self.remaining_attributes = remaining_attributes
        self.classification = node_classification(class_values)
        self.entropy = entropy (class_values)
        self.class_values = class_values
        self.pure_leaf = check_if_pure_leaf (class_values)

    def get_weighted_conditional_entropy(self, total_vals):
        if total_vals == 0:
            return 0
        return self.entropy * (self.total_classes / total_vals)

def check_if_pure_leaf(class_values):
    class_with_zero_vals = 0
    for key, val in class_values.items():
        if val == 0:
            class_with_zero_vals += 1

    # if two out of three classes have 0 values than leaf node is pure
    return (class_with_zero_vals == 2)
    
def entropy (class_values):
    total = 0

    for key, value in class_values.items():
        total += value

    if total == 0:
        return 0

    total_entropy = 0
    for key, value in class_values.items():
        total_entropy += (value / total) * get_log_base2 (value / total)

    return (total_entropy * -1)
        
def get_log_base2 (value):
    if value in [0, 1]:
        return 0
    
    return math.log2(value)

def node_classification(class_values):
    max_class = 0
    classification = None
    for key, value in class_values.items():
        if value > max_class:
            max_class = value
            classification = key

    return classification or 0

def count_distinct_vals(rows):
    counter = {0:0, 1:0, 2:0}
    for row in rows:
        counter[int(row)] += 1

    return counter

def get_column_rows(input_arr, column):
    rows = []
    for row in input_arr:
        rows.append (row[column])
    
    return rows

def get_count_and_targeted_rows (rows, target_val, parents_targeted_rows):
    targeted_rows = []
    counter = 0
    for idx in parents_targeted_rows:
        row = rows [idx]
        if int (row) == target_val:
            counter += 1
            targeted_rows.append (idx)

    return counter, targeted_rows

def get_restricted_distinct_class_vals (rows, targeted_rows):
    counter = {0:0, 1:0, 2:0}
    for idx, row in enumerate (rows):
        if idx in targeted_rows:
            counter[int(row)] += 1

    return counter

def display_tree (entry_node, tab_multiplier):
    if entry_node == None:
        return

    if entry_node.childs == None:
        return

    for idx, child in enumerate (entry_node.childs):
        leaf_node = child.childs == None
        str_to_print = ("| " * tab_multiplier) + entry_node.attribute_value + " = %s"%idx + " : " + (leaf_node * str (child.classification))
        print (str_to_print)

        display_tree (child, tab_multiplier + 1)

def build_tree (node, training_data, class_values):
    if len (node.remaining_attributes) == 0 or node.pure_leaf == True:
        return

    information_gain = 0
    headers = node.remaining_attributes

    target_attribute = None
    target_child_node = None

    for idx, attribute in enumerate (headers):
        target_arr = training_data [attribute]
        parents_targeted_rows = node.targeted_rows
        node_array = []
        # get targeted attribute value, we already know this {0, 1, 2}
        for val in [0, 1, 2]:
            counter, targeted_rows = get_count_and_targeted_rows(target_arr, val, parents_targeted_rows)
            res_class_vals = get_restricted_distinct_class_vals (class_values, targeted_rows)
            new_set = copy.deepcopy(node.remaining_attributes)
            new_set.remove(attribute)
            new_node = Node (res_class_vals, targeted_rows, new_set, counter)
            node_array.append (new_node)

        # calculate Information Gain and select the node with the highest one
        total_vals = node.total_classes
        information_gain_temp = node.entropy - (node_array[0].get_weighted_conditional_entropy(total_vals) + node_array[1].get_weighted_conditional_entropy(total_vals) + node_array[2].get_weighted_conditional_entropy(total_vals))
        if information_gain_temp > information_gain:
            information_gain = information_gain_temp
            target_child_node = node_array
            target_attribute = attribute

    node.childs = target_child_node
    node.attribute_value = target_attribute

    if node.childs == None:
        return

    for child in node.childs:
        build_tree (child, training_data, class_values)


def classify (row, decision_tree, headers):
    if decision_tree.childs == None:
        return decision_tree.classification

    attribute = decision_tree.attribute_value
    index = headers.index(attribute)
    row_val = row [index]
    child = decision_tree.childs[int (row_val)]
    
    return classify (row, child, headers)

def check_accuracy (training_data, decision_tree, headers):
    class_indx = len (headers)
    correct_classifications = 0
    for row in training_data:
        classification = classify (row, decision_tree, headers)
        actual_class_value = row [class_indx]
        if classification == int (actual_class_value):
            correct_classifications += 1

    return (correct_classifications / len (training_data)) * 100


def create_learning_curve_set (data, test_data, headers):
    learning_curve_set = []
    data_partition = 100
    while data_partition < 800:
        data_copy = copy.deepcopy (data)
        restricted_data_copy = data_copy[0:data_partition]
        ind = len (data[0]) - 1
        columnn_arr = {}
        for idx, header in enumerate (headers):
            column_array = get_column_rows (restricted_data_copy, idx)
            columnn_arr[header] = column_array

        class_rows = []
        for row in restricted_data_copy:
            class_rows.append (row[ind])

        total_rows = len (restricted_data_copy)
        class_vals = count_distinct_vals(class_rows)

        root = Node (class_vals, list (range(0,total_rows)), headers, total_rows)
        build_tree(root, columnn_arr, class_rows)
        test_accuracy = check_accuracy (test_data, root, headers)
        learning_curve_set.append ((data_partition, test_accuracy))
        data_partition += 100

    print (learning_curve_set)

# ####################################
#
# Running the Application proceduraly
#
# #####################################

if len (sys.argv) != 3:
    raise ValueError("Please provide two arguments")

training_set_file_path = sys.argv [1]
test_set_file_path = sys.argv [2]

input_arr = [i.strip().split() for i in open(training_set_file_path).readlines()]
test_set = [i.strip().split() for i in open(test_set_file_path).readlines()]

ind = len (input_arr[0]) - 1
headers = input_arr [0][0:ind]
data = input_arr [1:]
columnn_arr = {}
for idx, header in enumerate (headers):
    column_array = get_column_rows (data, idx)
    columnn_arr[header] = column_array

class_rows = []
for row in data:
    class_rows.append (row[ind])

total_rows = len (data)
class_vals = count_distinct_vals(class_rows)

root = Node (class_vals, list (range(0,total_rows)), headers, total_rows)

build_tree(root, columnn_arr, class_rows)
display_tree (root, 0)

training_accuracy = math.ceil (check_accuracy (data, root, headers) * 10) / 10
print ("Accuracy on training set(%s"%total_rows + " instances) = %s"%training_accuracy + "%")

test_data = test_set [1:]
total_test_rows = len (test_data)
test_accuracy = math.ceil (check_accuracy (test_data, root, headers) * 10) / 10
print ("Accuracy on test set(%s"%total_test_rows + " instances) = %s"%test_accuracy + "%")

create_learning_curve_set (data, test_data, headers)