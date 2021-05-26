import networkx as nx
import numpy as np
import pdb
import gzip
import csv
import pandas as pd
import os
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict as dd, OrderedDict
from haversine import haversine
import sys
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import PatchCollection
import pickle

import copy
import numpy as np


import pdb

class KDTree:
    def __init__(self, bucket_size, dimensions, parent=None):
        self.bucket_size = bucket_size
        self.parent = None
        self.left = None
        self.right = None
        self.split_dimension = None
        self.split_value = None
        self.index_locations = []
        self.location_count = 0
        self.min_limit = [np.Inf] * dimensions
        self.max_limit = [-np.Inf] * dimensions
        self.dimensions = dimensions

    def get_leaf(self, location):
        if not self.left and not self.right:
            return self
        elif location[self.split_dimension] <= self.split_value:
            return self.left.get_leaf(location)
        else:
            return self.right.get_leaf(location)

    def add_point(self, index_location_tuple):
        self.index_locations.append(index_location_tuple)
        self.location_count += 1
        self.extendBounds(index_location_tuple[1])
        self.min_boundary = copy.deepcopy(self.min_limit)
        self.max_boundary = copy.deepcopy(self.max_limit)

    def extendBounds(self, location):
        #empty
        if self.min_limit == None:
            self.min_limit = copy.deepcopy(location)
            self.max_limit = copy.deepcopy(location)
            return
        for i in range(self.dimensions):
            self.min_limit[i] = min(self.min_limit[i], location[i])
            self.max_limit[i] = max(self.max_limit[i], location[i])

    def findWidestAxis(self):
        widths = [self.max_limit[i] - self.min_limit[i] for i in range(self.dimensions)]
        widest_axis = np.argmax(widths)
        return widest_axis
    def getNodes(self):
        nodes = []
        self.getNodesHelper(nodes)
        return nodes

    def getNodesHelper(self, nodes):
        nodes.append(self)
        if self.left:
            self.left.getNodesHelper(nodes)
        if self.right:
            self.right.getNodesHelper(nodes)

    def getLeaves(self):
        leaves = []
        self.getLeavesHelper(leaves)
        return leaves

    def getLeavesHelper(self, leaves):
        if not self.right and not self.left:
            leaves.append(self)
        else:
            if self.left:
                self.left.getLeavesHelper(leaves)
            if self.right:
                self.right.getLeavesHelper(leaves)

    def balance(self):
        self.nodeSplit(self)

    def nodeSplit(self, cursor, empty_non_leaf=True):
        if cursor.location_count > cursor.bucket_size:
            cursor.split_dimension = cursor.findWidestAxis()
            #the partition method is the median of all values in the widest dimension
            cursor.split_value = np.median([cursor.index_locations[i][1][cursor.split_dimension] for i in range(cursor.location_count)])
            # if width is 0 (all the values are the same) don't partition
            if cursor.min_limit[cursor.split_dimension] == cursor.max_limit[cursor.split_dimension]:
                return
            # Don't let the split value be the same as the upper value as
            # can happen due to rounding errors!
            if cursor.split_value == cursor.max_limit[cursor.split_dimension]:
                cursor.split_value = cursor.min_limit[cursor.split_dimension]
            cursor.left = KDTree(bucket_size=cursor.bucket_size, dimensions=cursor.dimensions, parent=cursor)
            cursor.right = KDTree(bucket_size=cursor.bucket_size, dimensions=cursor.dimensions, parent=cursor)

            cursor.left.min_boundary = copy.deepcopy(cursor.min_boundary)
            cursor.left.max_boundary = copy.deepcopy(cursor.max_boundary)
            cursor.right.min_boundary = copy.deepcopy(cursor.min_boundary)
            cursor.right.max_boundary = copy.deepcopy(cursor.max_boundary)
            cursor.left.max_boundary[cursor.split_dimension] = cursor.split_value
            cursor.right.min_boundary[cursor.split_dimension] = cursor.split_value

            for index_loc in cursor.index_locations:
                if index_loc[1][cursor.split_dimension] > cursor.split_value:
                    cursor.right.index_locations.append(index_loc)
                    cursor.right.location_count += 1
                    cursor.right.extendBounds(index_loc[1])
                else:
                    cursor.left.index_locations.append(index_loc)
                    cursor.left.location_count += 1
                    cursor.left.extendBounds(index_loc[1])
            if empty_non_leaf:
                cursor.index_locations = []
            cursor.nodeSplit(cursor.left)
            cursor.nodeSplit(cursor.right)


class KDTreeClustering():
    def __init__(self, bucket_size=10):
        self.bucket_size = bucket_size
        self.is_fitted = False

    def fit(self, X):
        #X is an array
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
            dimensions = X.shape[1]
        else:
            n_samples = len(X)
            dimensions = len(X[0])

        self.kdtree = KDTree(bucket_size=self.bucket_size, dimensions=dimensions, parent=None)
        for i in range(n_samples):
            self.kdtree.add_point((i, X[i]))
        self.kdtree.nodeSplit(cursor=self.kdtree, empty_non_leaf=True)
        self.clusters = [leave.index_locations for leave in self.kdtree.getLeaves()]
        clusters = [cluster.index_locations for cluster in self.kdtree.getLeaves()]
        results = np.zeros((n_samples,), dtype=int)
        for i, id_locs in enumerate(clusters):
            for id, l in id_locs:
                results[id] = i
        self.clusters = results
        self.num_clusters = len(clusters)
        self.is_fitted = True

    def get_clusters(self):
        if self.is_fitted:
            return self.clusters

#from networkx.algorithms.bipartite.projection import weighted_projected_graph
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def dump_obj(obj, filename, protocol=-1, serializer=pickle):
    with gzip.open(filename, 'wb') as fout:
        serializer.dump(obj, fout, protocol)
def load_obj(filename, serializer=pickle):
    with gzip.open(filename, 'rb') as fin:
        obj = serializer.load(fin)
    return obj



def efficient_collaboration_weighted_projected_graph2(B, nodes):
    nodes = set(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    all_nodes = set(B.nodes())
    i = 0
    tenpercent = len(all_nodes) / 10
    for m in all_nodes:
        if i % tenpercent == 0:
            logging.info(str(10 * i / tenpercent) + "%")
        i += 1

        nbrs = B[m]
        target_nbrs = [t for t in nbrs if t in nodes]
        if m in nodes:
            for n in target_nbrs:
                if m < n:
                    if not G.has_edge(m, n):
                        G.add_edge(m, n)
        for n1 in target_nbrs:
            for n2 in target_nbrs:
                if n1 < n2:
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2)
    return G

class DataLoader():
    def __init__(self, data_home, bucket_size=50, encoding='latin',
                 celebrity_threshold=10, one_hot_labels=False, mindf=10, maxdf=0.2,
                 norm='l2', idf=True, btf=True, tokenizer=None, subtf=False, stops=None,
                 token_pattern=r'(?u)(?<![#@])\b\w\w+\b', vocab=None):
        self.data_home = data_home
        self.bucket_size = bucket_size
        self.encoding = encoding
        self.celebrity_threshold = celebrity_threshold
        self.one_hot_labels = one_hot_labels
        self.mindf = mindf
        self.maxdf = maxdf
        self.norm = norm
        self.idf = idf
        self.btf = btf
        self.tokenizer = tokenizer
        self.subtf = subtf
        self.stops = stops if stops else 'english'
        self.token_pattern = token_pattern
        self.vocab = vocab
        self.biggraph = None

    def load_data(self):
        logging.info('loading the dataset from %s' %self.data_home)
        train_file = os.path.join(self.data_home, 'user_info.train.gz')
        dev_file = os.path.join(self.data_home, 'user_info.dev.gz')
        test_file = os.path.join(self.data_home, 'user_info.test.gz')

        df_train = pd.read_csv(train_file, delimiter='\t', encoding='latin', names=['user', 'lat', 'lon', 'text'], quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_dev = pd.read_csv(dev_file, delimiter='\t', encoding='latin', names=['user', 'lat', 'lon', 'text'], quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_test = pd.read_csv(test_file, delimiter='\t', encoding='latin', names=['user', 'lat', 'lon', 'text'], quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_train.dropna(inplace=True)
        df_dev.dropna(inplace=True)
        df_test.dropna(inplace=True)
        df_train['user'] = df_train['user'].apply(lambda x: str(x).lower())
        df_train.drop_duplicates(['user'], inplace=True, keep='last')
        df_train.set_index(['user'], drop=True, append=False, inplace=True)
        df_train.sort_index(inplace=True)
        df_dev['user'] = df_dev['user'].apply(lambda x: str(x).lower())
        df_dev.drop_duplicates(['user'], inplace=True, keep='last')
        df_dev.set_index(['user'], drop=True, append=False, inplace=True)
        df_dev.sort_index(inplace=True)
        df_test['user'] = df_test['user'].apply(lambda x: str(x).lower())
        df_test.drop_duplicates(['user'], inplace=True, keep='last')
        df_test.set_index(['user'], drop=True, append=False, inplace=True)
        df_test.sort_index(inplace=True)
        self.df_train = df_train
        self.df_dev = df_dev
        self.df_test = df_test


    def get_graph(self):
        g = nx.Graph()
        nodes = set(self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist())
        assert len(nodes) == len(self.df_train) + len(self.df_dev) + len(self.df_test), 'duplicate target node'
        nodes_list = self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist()
        node_id = {node:id for id, node in enumerate(nodes_list)}
        g.add_nodes_from(node_id.values())
        for node in nodes:
            g.add_edge(node_id[node], node_id[node])
        pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        pattern = re.compile(pattern)
        logging.info('adding the train graph')
        for i in range(len(self.df_train)):
            user = self.df_train.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_train.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        logging.info('adding the dev graph')
        for i in range(len(self.df_dev)):
            user = self.df_dev.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_dev.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        logging.info('adding the test graph')
        for i in range(len(self.df_test)):
            user = self.df_test.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_test.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        celebrities = []
        for i in range(len(nodes_list), len(node_id)):
            deg = len(g[i])
            if deg == 1 or deg > self.celebrity_threshold:
                celebrities.append(i)
        logging.info('removing %d celebrity nodes with degree higher than %d' % (len(celebrities), self.celebrity_threshold))
        g.remove_nodes_from(celebrities)
        self.biggraph = g
        logging.info('projecting the graph')
        projected_g = efficient_collaboration_weighted_projected_graph2(g, range(len(nodes_list)))
        logging.info('#nodes: %d, #edges: %d' %(nx.number_of_nodes(projected_g), nx.number_of_edges(projected_g)))
        self.graph = projected_g

    def get_graph_temp(self):
        from haversine import haversine
        from collections import defaultdict
        g = nx.Graph()
        nodes = set(self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist())
        assert len(nodes) == len(self.df_train) + len(self.df_dev) + len(self.df_test), 'duplicate target node'
        nodes_list = self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist()
        node_id = {node:id for id, node in enumerate(nodes_list)}
        g.add_nodes_from(node_id.values())
        train_locs = self.df_train[['lat', 'lon']].values
        for node in nodes:
            g.add_edge(node_id[node], node_id[node])
        pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        pattern = re.compile(pattern)
        logging.info('adding the train graph')
        for i in range(len(self.df_train)):
            user = self.df_train.index[i]
            user_id = node_id[user]
            mentions = [m for m in pattern.findall(self.df_train.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        celebrities = []
        for i in range(len(nodes_list), len(node_id)):
            deg = len(g[i])
            if deg > self.celebrity_threshold:
                celebrities.append(i)
        #get neighbours of celebrities
        id_node = {v:k for k, v in node_id.iteritems()}

        degree_distmean = defaultdict(list)
        degree_distance = defaultdict(list)
        c_distmean = {}
        for c in celebrities:
            c_name = id_node[c]
            c_nbrs = g[c].keys()
            c_degree = len(c_nbrs)
            c_locs = train_locs[c_nbrs, :]
            c_lats = c_locs[:, 0]
            c_lons = c_locs[:, 1]
            c_median_lat = np.median(c_lats)
            c_median_lon = np.median(c_lons)
            distances = [haversine((c_median_lat, c_median_lon), tuple(c_locs[i].tolist())) for i in range(c_locs.shape[0])]
            degree_distance[c_degree].extend(distances)
            c_meandist = np.mean(distances)
            degree_distmean[c_degree].append(c_meandist)
            c_distmean[c_name] = [c_degree, c_meandist]
        with open('celebrity.pkl', 'wb') as fin:
            pickle.dump((c_distmean, degree_distmean, degree_distance), fin)


        logging.info('removing %d celebrity nodes with degree higher than %d' % (len(celebrities), self.celebrity_threshold))

        self.biggraph = g

    def longest_path(self, g):
        nodes = g.nodes()
        pathlen_counter = Counter()
        for n1 in nodes:
            for n2 in nodes:
                if n1 < n2:
                    for path in nx.all_simple_paths(g, source=n1, target=n2):
                        pathlen = len(path)
                        pathlen_counter[pathlen] += 1
        return pathlen_counter
    def tfidf(self):
        #keep both hashtags and mentions
        #token_pattern=r'(?u)@?#?\b\w\w+\b'
        #remove hashtags and mentions
        #token_pattern = r'(?u)(?<![#@])\b\w+\b'
        #just remove mentions and remove hashsign from hashtags
        #token_pattern = r'(?u)(?<![@])\b\w+\b'
        #remove mentions but keep hashtags with their sign
        #token_pattern = r'(?u)(?<![@])#?\b\w\w+\b'
        #remove multple occurrences of a character after 2 times yesss => yess
        #re.sub(r"(.)\1+", r"\1\1", s)
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, token_pattern=self.token_pattern, use_idf=self.idf,
                                    norm=self.norm, binary=self.btf, sublinear_tf=self.subtf,
                                    min_df=self.mindf, max_df=self.maxdf, ngram_range=(1, 1), stop_words=self.stops,
                                     vocabulary=self.vocab, encoding=self.encoding, dtype='float32')
        logging.info(self.vectorizer)
        self.X_train = self.vectorizer.fit_transform(self.df_train.text.values)
        self.X_dev = self.vectorizer.transform(self.df_dev.text.values)
        self.X_test = self.vectorizer.transform(self.df_test.text.values)
        logging.info("training    n_samples: %d, n_features: %d" % self.X_train.shape)
        logging.info("development n_samples: %d, n_features: %d" % self.X_dev.shape)
        logging.info("test        n_samples: %d, n_features: %d" % self.X_test.shape)

    def assignClasses(self):
        clusterer = KDTreeClustering(bucket_size=self.bucket_size)
        train_locs = self.df_train[['lat', 'lon']].values
        clusterer.fit(train_locs)
        clusters = clusterer.get_clusters()
        cluster_points = dd(list)
        for i, cluster in enumerate(clusters):
            cluster_points[cluster].append(train_locs[i])
        logging.info('#labels: %d' %len(cluster_points))
        self.cluster_median = OrderedDict()
        for cluster in sorted(cluster_points):
            points = cluster_points[cluster]
            median_lat = np.median([p[0] for p in points])
            median_lon = np.median([p[1] for p in points])
            self.cluster_median[cluster] = (median_lat, median_lon)
        dev_locs = self.df_dev[['lat', 'lon']].values
        test_locs = self.df_test[['lat', 'lon']].values
        nnbr = NearestNeighbors(n_neighbors=1, algorithm='brute', leaf_size=1, metric=haversine, n_jobs=4)
        nnbr.fit(np.array([v for v in self.cluster_median.values()]))
        self.dev_classes = nnbr.kneighbors(dev_locs, n_neighbors=1, return_distance=False)[:, 0]
        self.test_classes = nnbr.kneighbors(test_locs, n_neighbors=1, return_distance=False)[:, 0]

        self.train_classes = clusters
        if self.one_hot_labels:
            num_labels = np.max(self.train_classes) + 1
            y_train = np.zeros((len(self.train_classes), num_labels), dtype=np.float32)
            y_train[np.arange(len(self.train_classes)), self.train_classes] = 1
            y_dev = np.zeros((len(self.dev_classes), num_labels), dtype=np.float32)
            y_dev[np.arange(len(self.dev_classes)), self.dev_classes] = 1
            y_test = np.zeros((len(self.test_classes), num_labels), dtype=np.float32)
            y_test[np.arange(len(self.test_classes)), self.test_classes] = 1
            self.train_classes = y_train
            self.dev_classes = y_dev
            self.test_classes = y_test
    def draw_kd_clusters2(self, filename, figsize=(4,3)):
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap, cm, maskoceans
        class KDTree:
            """Simple KD tree class"""

            # class initialization function
            def __init__(self, data, mins, maxs):
                self.data = np.asarray(data)

                # data should be two-dimensional
                assert self.data.shape[1] == 2

                if mins is None:
                    mins = data.min(0)
                if maxs is None:
                    maxs = data.max(0)

                self.mins = np.asarray(mins)
                self.maxs = np.asarray(maxs)
                self.sizes = self.maxs - self.mins

                self.child1 = None
                self.child2 = None

                if len(data) > 1:
                    # sort on the dimension with the largest spread
                    largest_dim = np.argmax(self.sizes)
                    i_sort = np.argsort(self.data[:, largest_dim])
                    self.data[:] = self.data[i_sort, :]

                    # find split point
                    N = self.data.shape[0]
                    split_point = 0.5 * (self.data[N / 2, largest_dim]
                                         + self.data[N / 2 - 1, largest_dim])

                    # create subnodes
                    mins1 = self.mins.copy()
                    mins1[largest_dim] = split_point
                    maxs2 = self.maxs.copy()
                    maxs2[largest_dim] = split_point

                    # Recursively build a KD-tree on each sub-node
                    self.child1 = KDTree(self.data[N / 2:], mins1, self.maxs)
                    self.child2 = KDTree(self.data[:N / 2], self.mins, maxs2)

            def draw_rectangle(self, ax, depth=None):
                """Recursively plot a visualization of the KD tree region"""
                if depth == 0:
                    rect = plt.Rectangle(self.mins, *self.sizes, ec='k', fc='none', lw=0.7)
                    ax.add_patch(rect)

                if self.child1 is not None:
                    if depth is None:
                        self.child1.draw_rectangle(ax)
                        self.child2.draw_rectangle(ax)
                    elif depth > 0:
                        self.child1.draw_rectangle(ax, depth - 1)
                        self.child2.draw_rectangle(ax, depth - 1)


        #------------------------------------------------------------
        # Create a set of structured random points in two dimensions
        np.random.seed(0)


        lllat = 24.396308
        lllon = -124.848974
        urlat =  49.384358
        urlon = -66.885444
        fig = plt.figure(figsize=figsize)
        m = Basemap(llcrnrlat=lllat,
        urcrnrlat=urlat,
        llcrnrlon=lllon,
        urcrnrlon=urlon,
        resolution='c', projection='cyl')
        m.drawmapboundary(fill_color = 'white')
        m.drawcoastlines(linewidth=0.4)
        m.drawcountries(linewidth=0.4)
        train_locs = self.df_train[['lon', 'lat']].values
        mlon, mlat = m(*(train_locs[:,1], train_locs[:,0]))
        train_locs = np.transpose(np.vstack((mlat, mlon)))
        ax = plt.gca()
        #fig = plt.figure()  # figsize=(4,4.2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        #------------------------------------------------------------
        # Use our KD Tree class to recursively divide the space
        KDT = KDTree(train_locs, [lllon-1, urlon+1], [lllat-1, urlat+1])

        #------------------------------------------------------------
        # Plot four different levels of the KD tree
        fig = plt.figure(figsize=figsize)
        '''
        fig.subplots_adjust(wspace=0.1, hspace=0.15,
                            left=0.1, right=0.9,
                            bottom=0.05, top=0.9)
        '''
        level = 8
        ax = plt.gca()
        #ax.scatter(X[:, 0], X[:, 1], s=9)
        KDT.draw_rectangle(ax, depth=level - 1)

        ax.set_xlim([-125, -60])  # pylab.xlim([-400, 400])
        ax.set_ylim([25, 50])

        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.yaxis.set_tick_params(size=0)
        ax.xaxis.set_tick_params(size=0)
        #plt.tick_params(axis='both', which='major', labelsize=25)
        #ax.labelsize = '25'
        #plt.subplots_adjust(bottom=0.2)
        m.drawlsmask(land_color='lightgray',ocean_color="#b0c4de", lakes=True)
        plt.tight_layout()
        plt.savefig(filename)

    def draw_kd_clusters(self, filename, figsize=(4,3)):
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap, cm, maskoceans
        #from matplotlib import style
        #import seaborn as sns
        #sns.set_style("white")
        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif')
        #plt.rcParams['axes.facecolor']='white'
        fig = plt.figure(figsize=figsize)

        lllat = 24.396308
        lllon = -124.848974
        urlat =  49.384358
        urlon = -66.885444
        m = Basemap(llcrnrlat=lllat,
        urcrnrlat=urlat,
        llcrnrlon=lllon,
        urcrnrlon=urlon,
        resolution='c', projection='cyl')
        m.drawmapboundary(fill_color = 'white')
        m.drawcoastlines(linewidth=0.2)
        m.drawcountries(linewidth=0.2)

        ax = plt.gca()
        #fig = plt.figure()  # figsize=(4,4.2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        clusterer = KDTreeClustering(bucket_size=self.bucket_size)
        train_locs = self.df_train[['lat', 'lon']].values
        mlon, mlat = m(*(train_locs[:,1], train_locs[:,0]))
        train_locs = np.transpose(np.vstack((mlat, mlon)))

        clusterer.fit(train_locs)
        clusters = clusterer.get_clusters()
        cluster_points = dd(list)
        for i, cluster in enumerate(clusters):
            cluster_points[cluster].append(train_locs[i])
        corners = []
        for i in clusters:
            points = np.vstack(cluster_points[i])
            min_lat, min_lon = points.min(axis=0)
            max_lat, max_lon = points.max(axis=0)
            min_lon, min_lat = m(min_lon, min_lat)
            max_lon, max_lat = m(max_lon, max_lat)
            corners.append([min_lat, min_lon, max_lat, max_lon])
        patches = []
        for corner in corners:
            min_lat, min_lon, max_lat, max_lon = corner
            rect = mpatches.Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat, facecolor=None, fill=False, linewidth=0.7)
            patches.append(rect)
        ax.add_collection(PatchCollection(patches))
        ax.set_xlim([-125, -60])  # pylab.xlim([-400, 400])
        ax.set_ylim([25, 50])

        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.yaxis.set_tick_params(size=0)
        ax.xaxis.set_tick_params(size=0)
        #plt.tick_params(axis='both', which='major', labelsize=25)
        #ax.labelsize = '25'
        #plt.subplots_adjust(bottom=0.2)
        m.drawlsmask(land_color='gray',ocean_color="#b0c4de", lakes=True)
        plt.tight_layout()
        plt.savefig(filename)
        #plt.close()
        print("the plot saved in " + filename)

    def draw_kmeans_clusters(self, filename, figsize=(4,3)):
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        from scipy.spatial import Voronoi, voronoi_plot_2d
        from mpl_toolkits.basemap import Basemap, cm, maskoceans
        #from matplotlib import style
        #import seaborn as sns
        #sns.set_style("white")
        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif')
        #plt.rcParams['axes.facecolor']='white'
        fig = plt.figure(figsize=figsize)
        lllat = 24.396308
        lllon = -124.848974
        urlat =  49.384358
        urlon = -66.885444
        m = Basemap(llcrnrlat=lllat,
        urcrnrlat=urlat,
        llcrnrlon=lllon,
        urcrnrlon=urlon,
        resolution='c', projection='cyl')
        m.drawmapboundary(fill_color = 'white')
        m.drawcoastlines(linewidth=0.2)
        m.drawcountries(linewidth=0.2)

        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for spine in ax.spines.itervalues():
            spine.set_visible(False)

        #fig = plt.figure()  # figsize=(4,4.2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        train_locs = self.df_train[['lat', 'lon']].values
        n_clusters = int(np.ceil(train_locs.shape[0] / self.bucket_size))
        n_clusters = 128
        logging.info('n_cluster %d' %n_clusters)
        clusterer = KMeans(n_clusters=n_clusters, n_jobs=10)
        clusterer.fit(train_locs)
        centroids = clusterer.cluster_centers_
        centroids[:,[0, 1]] = centroids[:,[1, 0]]
        mlon, mlat = m(*(centroids[:,0], centroids[:,1]))
        centroids = np.transpose(np.vstack((mlon, mlat)))

        vor = Voronoi(centroids)


        #ax.set_xlim([-125, -60])  # pylab.xlim([-400, 400])
        #ax.set_ylim([25, 50])

        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.yaxis.set_tick_params(size=0)
        ax.xaxis.set_tick_params(size=0)
        #plt.tick_params(axis='both', which='major', labelsize=25)
        #ax.labelsize = '25'
        #plt.subplots_adjust(bottom=0.2)
        voronoi_plot_2d(vor, show_points=False, show_vertices=False, ax=ax, line_width=0.7)
        m.drawlsmask(land_color='lightgray',ocean_color="#b0c4de", lakes=True)
        plt.tight_layout()
        plt.savefig(filename)
        #plt.close()
        print("the plot saved in " + filename)

    def draw_training_points(self, filename='points.pdf', world=False, figsize=(4,3)):
        '''
        draws training points on map
        '''
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap, cm, maskoceans


        fig = plt.figure(figsize=figsize)
        lllat = 24.396308
        lllon = -124.848974
        urlat =  49.384358
        urlon = -66.885444
        if world:
            lllat = -90
            lllon = -180
            urlat = 90
            urlon = 180
        m = Basemap(llcrnrlat=lllat,
        urcrnrlat=urlat,
        llcrnrlon=lllon,
        urcrnrlon=urlon,
        resolution='c', projection='cyl')
        m.drawmapboundary(fill_color = 'white')
        m.drawcoastlines(linewidth=0.2)
        m.drawcountries(linewidth=0.2)

        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for spine in ax.spines.itervalues():
            spine.set_visible(False)

        #fig = plt.figure()  # figsize=(4,4.2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        train_locs = self.df_train[['lat', 'lon']].values
        mlon, mlat = m(*(train_locs[:,1], train_locs[:,0]))
        #m.scatter(mlon, mlat, color='red', s=0.6)
        m.plot(mlon, mlat, 'r.', markersize=1)
        m.drawlsmask(land_color='lightgray',ocean_color="#b0c4de", lakes=True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print("the plot saved in " + filename)


if __name__ == '__main__':
    data_loader = DataLoader(data_home='./data/', dataset='cmu')
    data_loader.load_data()
    data_loader.get_graph()
    data_loader.tfidf()
    data_loader.assignClasses()
    pdb.set_trace()
