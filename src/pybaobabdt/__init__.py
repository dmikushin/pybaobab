# Package imports

from sklearn import tree
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier 
from random import randint
import numpy as np
import pygraphviz as pgv
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from collections import defaultdict
from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import random
import copy

def convert(data):
    mapping = dict()
    convertedAttributes = []
    number = preprocessing.LabelEncoder()
    for c in data.columns:
        if c == 'class':
            continue
        # If column is not numeric
        if not is_numeric_dtype(data[c]):
            convertedAttributes.append(c)
            data[c] = number.fit_transform(data[c])
            map = dict(zip(number.transform(number.classes_), number.classes_))
            mapping[c] = map
    data = data.fillna(-999)    
    
    return data, mapping, convertedAttributes
    

def linewidth_from_data_units(linewidth, axis, reference='y'):
    """
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return linewidth * (length / value_range)




def rect(x,y,width,height,color):
    y = y - height # define rectangle by xy = topleft
    patch = patches.Rectangle(xy=(x,y), width=width, height=height, angle=0, color=color, alpha=0.5)
    return patch

def addPath(points, linewidth, color, alpha=0.5):
    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             ]   
    
    path = Path(points, codes) 
        
    patch = patches.PathPatch(path, facecolor='none', lw=linewidth, alpha = alpha, edgecolor=color, antialiased=True, capstyle='butt', snap=False)
    return patch


def drawTree(decisionTreeClassifier, features=[], model=[], colormap='viridis', size=15, dpi=300, ratio=1, classes=[], maxdepth=-1, ax=-1, ):
    clf = decisionTreeClassifier
    sizeX = size
    sizeY = size
    sizeDpi = dpi
    
    if model == []:
        model = clf
    
    #class_names = df['class'].unique()
    #class_names = clf.classes_
    class_names = copy.deepcopy(model.classes_)
    #print('Original:', class_names)
    #print(class_names)
    #originalNames = clf.classes_
    #print(originalNames)
    #random.shuffle(class_names)
    #print('New:', class_names)
    #keys = list(range(0,len(class_names)))
    #print(keys)
    # values = []
    # for i in class_names:
    #     values.append(list(clf.classes_).index(i))
    #print(values)
    #mapClasses = dict(zip(keys,values))
    #mapClasses_r = dict(zip(values, keys))
    #print('Mapping:', mapClasses)

    nClasses = len(class_names) #len(dataset.target_names)
    if isinstance(colormap, str):
        colors = (plt.cm.get_cmap(colormap, nClasses)) #Spectral, Blues, viridis
    else:
        colors = colormap

    optimal_ordering = (classes == [])
    
    if not optimal_ordering:
        keys   = list(range(0,nClasses))
        
        #keys = classes
        originalOrdering = []
        for c in list(model.classes_):
            originalOrdering.append(c)
        
        #print(originalOrdering)
        
        keys = [originalOrdering.index(i) for i in classes]
        values = list(range(0,nClasses))

        mapClasses = dict(zip(keys,values))
        mapClasses_r = dict(zip(values, keys))

        for i in range(0,nClasses):
            class_names[i] = clf.classes_[mapClasses[i]]    
    
    
    
    #print(class_names)

    #dataset.feature_names
    DTree = clf.tree_

    n_nodes = DTree.node_count
    children_left = DTree.children_right
    children_right = DTree.children_left
    feature = DTree.feature
    threshold = DTree.threshold

    feature_name = [
        features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in DTree.feature
    ]

    min_Node_size = 0.5
    max_Node_size = 6.0

    min_items = 0.0
    max_items = float(np.sum(DTree.value[0]))
    #print('max_items', max_items)

    g = pgv.AGraph(strict=True, directed=True)

    dpi = 72.0

    g.graph_attr['rankdir']  = 'TB'
    g.graph_attr['ratio']    = ratio #0.75
    g.node_attr['shape']     = 'box'
    g.node_attr['fixedsize'] = True
    g.graph_attr['ranksep']  = 2    #0.75, default 0.5

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1, -1)]  # seed is the root node id and its parent depth
    parent = {}
    childIndex = {}
    while len(stack) > 0:
        node_id, parent_depth, parent_node_id = stack.pop()
        node_depth[node_id] = parent_depth + 1
        
        g.add_node(node_id, label='')       
        
        n = g.get_node(node_id)
        n_items = np.sum(DTree.value[node_id])
        s = (((n_items) / (max_items)) * max_Node_size) * 1.5

        n.attr['width']  = s
        n.attr['height'] = s         
        
    #    minimum_len = 1
    #     if (parent_node_id == 0):
    #         minimum_len = 0.5*s

        if (node_id != 0):
            if (parent_node_id != 0):
                g.add_edge(parent_node_id, node_id, weight=n_items) #, minlen=minimum_len)
            else:
                g.add_edge(parent_node_id, node_id, weight=n_items, minlen=s)
            parent[node_id] = parent_node_id
          
        if maxdepth != -1 and node_depth[node_id] >= maxdepth:
            is_leaves[node_id] = True
            continue
        
        if (children_right[node_id] != children_left[node_id]):
                stack.append((children_left[node_id], parent_depth + 1, node_id))        
                stack.append((children_right[node_id], parent_depth + 1, node_id))
        else:
            is_leaves[node_id] = True

    g.layout(prog='dot') #dot, twopi
    #g.draw('testTree.png')


    size = (sizeX,sizeY)
    
    if ax == -1:

        plt.rcParams['figure.figsize'] = size   
        fig = plt.figure(figsize=size, dpi=sizeDpi)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    
        
    nodes=g.nodes()

    xs = []
    ys = []
    nodeInfo = {}
    xPosClasses = []
    classesX = []
    for c in range(0,nClasses):
        xPosClasses.append(0)
        classesX.append([])
    #totalItems = np.sum(DTree.value[0][0])


    for n in nodes:
        x = float(n.attr['pos'].split(',')[0])
        y = float(n.attr['pos'].split(',')[1])

        width  = float(n.attr['width']) * dpi
        height = float(n.attr['height']) * dpi

        x = x - 0.5*width
        y = y + 0.5*height

        xs.append(x)
        xs.append(x+width)
        ys.append(y)
        ys.append(y-height)

        nodeInfo[int(n)] = (x,y,width,height)

        nodeId = int(n)
        if is_leaves[nodeId]:
            idxClass = np.argmax(DTree.value[nodeId][0])
            nItems = np.sum(DTree.value[nodeId][0][idxClass])
            totalItems = DTree.value[0][0][idxClass]
            xPosClasses[idxClass] += float(nItems)/totalItems * x #Take arithmetic mean
            #classesX[idxClass].append(x)

    # Determine 'optimal' ordering 
    if optimal_ordering:
        keys   = list(range(0,nClasses))
        values = xPosClasses
        idxPosClasses = dict(zip(keys, values))
        sorted_by_value = sorted(idxPosClasses.items(), key=lambda kv: kv[1])
        #print(idxPosClasses)
        #print(sorted_by_value)
        keys = [i[0] for i in sorted_by_value]
        #print(keys)
        values = list(range(0,nClasses))

        mapClasses = dict(zip(keys,values))
        mapClasses_r = dict(zip(values, keys))

        for i in range(0,nClasses):
            class_names[i] = clf.classes_[mapClasses[i]]

    #print('NEW:', class_names)    
    
    # Make sure root node is in center (weighted arithmetic mean) of its children
    rootNode = 0
    totalItems = np.sum(DTree.value[rootNode])
    x,y, width, height = nodeInfo[rootNode]
    allChildren = []
    allChildren.append(children_left[rootNode])
    allChildren.append(children_right[rootNode])
    xPos = []
    for child in allChildren:
        childItems = np.sum(DTree.value[child])
        ratio = float(childItems) / totalItems
        xPos.append(ratio * nodeInfo[child][0])
    nodeInfo[0] = (np.sum(xPos), y, width, height)

    
    absmin = min(np.min(xs), np.min(ys))
    absmax = max(np.max(xs), np.max(ys))

    ax.set_xlim(absmin, absmax)
    ax.set_ylim(absmin, absmax)
    ax.set_aspect('equal', 'datalim')

    pl = 500
    lineW = linewidth_from_data_units(pl, ax, reference='x')

    for n in nodes:
        n = int(n)
        if is_leaves[n]:

            x = nodeInfo[n][0]
            y = nodeInfo[n][1]
            w = nodeInfo[n][2]
            h = nodeInfo[n][3]

            c = np.argmax(DTree.value[n][0])
            #print(DTree.value[n][0])
            #print(c)

            ax.add_patch(rect(x,y,w,h,colors(mapClasses_r[c]/nClasses)))

    edges = g.edges()
    nodeEdgesXTop    = defaultdict(list)
    nodeEdgesXBottom = defaultdict(list)
    nodeEdgesWidthT  = defaultdict(list)
    nodeEdgesWidthB  = defaultdict(list)
    for e in edges:
        fromN = int(e[0])
        toN   = int(e[1])

        fx = nodeInfo[fromN][0]
        fy = nodeInfo[fromN][1]
        tx = nodeInfo[toN][0]
        ty = nodeInfo[toN][1]

        fw = nodeInfo[fromN][2]
        fh = nodeInfo[fromN][3]    
        tw = nodeInfo[toN][2]
        th = nodeInfo[toN][3]   

        if children_right[fromN] == toN:
            x1 = fx+0.5*tw
        else:
            x1 = fx+fw-0.5*tw


        y1 = fy-fh

        x2 = tx+0.5*tw
        y2 = ty

        rh = y1-y2

        p1 = [
                (x1, y1),
                (x1, y1-0.5*rh),
                (x2, y1-0.5*rh),
                (x2, y2),
             ]

    
        # For each edge we need to draw #classes splines
        startX  = x1 - 0.5*tw
        startX2 = x2 - 0.5*tw
        for c in range(0,nClasses):
            cItems   = float(DTree.value[toN][0][mapClasses[c]])
            if cItems == 0:
                nodeEdgesXTop[toN].append(-1)
                nodeEdgesXBottom[fromN].append(-1)
                nodeEdgesWidthT[toN].append(-1)
                nodeEdgesWidthB[fromN].append(-1)            
                continue
            mItems = np.sum(DTree.value[toN][0])
            perC = cItems / mItems
            w = perC * tw
            xn1 = startX + 0.5* w
            xn2 = startX2 + 0.5* w
            p1 = [
                    (xn1, y1),
                    (xn1, y1-0.5*rh),
                    (xn2, y1-0.5*rh),
                    (xn2, y2),
                 ]
            lineW = linewidth_from_data_units(w, ax, reference='x')
            ax.add_patch(addPath(p1, lineW, colors(c/nClasses)))

            nodeEdgesXTop[toN].append(xn2)
            nodeEdgesXBottom[fromN].append(xn1)
            nodeEdgesWidthT[toN].append(w)
            nodeEdgesWidthB[fromN].append(w)

            startX   += w
            startX2  += w


    # Deal with root node separately
    previousClassWidthN = defaultdict(float)
    startClassX = defaultdict(float)

    startX = nodeInfo[0][0]
    tw = nodeInfo[0][2]
    for c in range(0,nClasses):
        cItems = float(DTree.value[0][0][mapClasses[c]])
        mItems = np.sum(DTree.value[0][0])
        perC = cItems / mItems
        w = perC * tw
        x1 = startX + 0.5*w

        nodeEdgesWidthT[0].append(w)
        nodeEdgesXTop[0].append(x1)

        startX += w

    # Draw splines on top of 'nodes'
    for n in nodes:
        n = int(n)

        x = nodeInfo[n][0]
        y = nodeInfo[n][1]
        w = nodeInfo[n][2]
        h = nodeInfo[n][3]

        previousClassWidth = defaultdict(float)
        for i in range(0,len(nodeEdgesXBottom[n])):
            if nodeEdgesXBottom[n][i] == -1:
                continue

            c = i % nClasses

            x1 = nodeEdgesXTop[n][c]
            x2 = nodeEdgesXBottom[n][i]
            w  = nodeEdgesWidthB[n][i]

            lw  = nodeEdgesWidthT[n][c]
            x1 = x1 - 0.5*lw + 0.5*w + previousClassWidth[c]

            previousClassWidth[c] = previousClassWidth[c] + w

            y1 = y
            y2 = y-h

            p1 = [
                    (x1, y1),
                    (x1, y1-0.5*h),
                    (x2, y1-0.5*h),
                    (x2, y2),
                 ]        

            lineW = linewidth_from_data_units(w, ax, reference='x')
            ax.add_patch(addPath(p1, lineW, colors(c/nClasses)))  

        nx = nodeInfo[n][0]
        ny = nodeInfo[n][1]
        nw = nodeInfo[n][2]
        nh = nodeInfo[n][3]

        name = feature_name[n]
        threshold = DTree.threshold[n]
        threshold = "{:.2f}".format(threshold)

        #if name != "undefined!":
        if not is_leaves[n]:

            fontSize = linewidth_from_data_units(min(nw,80), ax, reference='y')
            lineSize = linewidth_from_data_units(15,  ax, reference='y')    
            feature = plt.annotate(name, xy=(nx+0.5*nw, ny-0.3*nh), color='black', size=fontSize, ha='center', va='center')
            feature.set_path_effects([path_effects.Stroke(linewidth=lineSize, foreground='white', alpha=0.5),
                                    path_effects.Normal()])


    #         if name in convertedAttributes:
    #             map = mapping[name]
    #             print(map)
    #             threshold = map[threshold]

            fontSize = linewidth_from_data_units(min(nw/2.0,50), ax, reference='y')
            threshold1 = plt.annotate('≤ ' + str(threshold), xy=(nx, ny-nh), color='black', size=fontSize, ha='right', va='center')
            threshold1.set_path_effects([path_effects.Stroke(linewidth=lineSize, foreground='white', alpha=0.5),
                                    path_effects.Normal()])

            threshold2 = plt.annotate('> ' + str(threshold), xy=(nx+nw, ny-nh), color='black', size=fontSize, ha='left', va='center')
            threshold2.set_path_effects([path_effects.Stroke(linewidth=lineSize, foreground='white', alpha=0.5),
                                    path_effects.Normal()])


    y = nodeInfo[0][1]
    for c in range(0,nClasses):  
        x = nodeEdgesXTop[0][c] - 0.25*nodeEdgesWidthT[0][c]
        fontSize = linewidth_from_data_units(min(55,nodeEdgesWidthT[0][c]), ax, reference='y')
        lineSize = linewidth_from_data_units(5,  ax, reference='y')    
        text = plt.annotate(str(class_names[c], 'utf-8'), xy=(x,y), color='black', size=fontSize, ha='left', va='bottom', rotation=45)
        #text = plt.annotate(str(class_names[c]), xy=(x,y), color='black', size=fontSize, ha='left', va='bottom', rotation=45)
        text.set_path_effects([path_effects.Stroke(linewidth=lineSize, foreground='white', alpha=0.5),
                                path_effects.Normal()])        


    ax.set_xlim(np.min(xs), np.max(xs))
    ax.set_ylim(np.min(ys), np.max(ys))
    ax.set_aspect('equal', 'datalim')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

    #plt.savefig('test003.png', format='png', dpi=1200)
    
    return ax