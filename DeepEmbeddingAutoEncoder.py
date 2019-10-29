#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:49:49 2017
@author: whoiles@ece.ubc.ca; fundamentalhoiles@gmail.com
"""
import os
#os.chdir('/wkdir/JMLR_Data')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
print(os.environ.get('KERAS_BACKEND'))
print(os.environ.get('CUDA_VISIBLE_DEVICES'))

#File->Settings->Project->Project Structure-> mark ImageAutoEncoders as a Source Folder
#%%
import numpy as np
import youtube.ytf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import LSTM, Concatenate, RepeatVector, TimeDistributed
from keras.models import Model
import keras.backend as K
from sklearn.cluster import KMeans
import re
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(reduce_len=True)
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatiser = WordNetLemmatizer()
from nltk import pos_tag
from keras.preprocessing import sequence
import pickle

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

########################################################################################################################
class TextMLParser:
    """
    https://spacy.io/usage/linguistic-features#named-entities
    Class for performing pre-processing of text for machine learning applications
    """

    def __init__(self, maxlength=None):
        """
        Initialization of the TextParser parameters
        """
        self.text = [] # Processed text of shape=(samples, maxlength, feature_dim)
        self.maxlength = 5 if maxlength is None else maxlength  # maximum sentence word length
        self.glove = None # Glove model
        # remove punctuation ( ?!,.:;`()-'" ), and replace with spaces
        # self.text = re.sub(r"[-?!,.:;'`()\"]", " ", self.text)

    def loadtext(self, text):
        '''Loads new text'''
        self.text = []
        self.text = text
        self.processtext()

    def processtext(self):
        '''Processes the raw text'''
        if bool(self.text):
            self.lemstem()
            self.text2glove()
            print('Text processing complete')
        else:
            print('Load text into class using TextMLParser.loadtext()')

    @staticmethod
    def lemmatize(string):
        '''sort words by grouping inflected or variant forms of the same word'''
        wordnet_tag = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}
        tokens = tknzr.tokenize(string)  # Tokenize
        postag = pos_tag(tokens)  # Part-Of-Speech Tags
        # Lemmatize or Stemming of text
        Xinput = [lemmatiser.lemmatize(postag[m][0], wordnet_tag.get(postag[m][1])) if wordnet_tag.get(
            postag[m][1]) != None else stemmer.stem(postag[m][0]) for m in range(len(postag))]
        return np.asarray([w.lower() for w in Xinput])

    def lemstem(self):
        '''lemmatize and stemming of text'''
        self.text = [self.lemmatize(w) for w in self.text]
        self.text = np.array(self.text)

    @staticmethod
    def loadGloveModel(gloveFile):
        '''Loads Glove text file word embedding vector
        wget nlp.stanford.edu/data/glove.twitter.27B.zip
        '''
        print('Loading Glove glove Model')
        f = open(gloveFile, 'r', encoding="utf8")
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print('Done with {} words loaded!'.format(str(len(model))))
        return model

    def text2glove(self):
        '''Converts text to numbers and pads to specific length'''
        if bool(self.glove):
            # word embedding
            self.text = np.array(
                [np.concatenate([self.glove.get(w, np.zeros(shape=(25,))).reshape((1, 25)) for w in s[:self.maxlength]])
                 for s in self.text])
            # pad with zero vectors
            self.text = [np.concatenate((A, np.zeros(shape=((self.maxlength - A.shape[0]), 25))), axis=0) for A in
                         self.text]
            # numpy array of shape (samples,maxlength,embeddinglength)
            self.text = np.asarray(self.text)
        else:
            self.glove = self.loadGloveModel('./Glove/glove.twitter.27B.25d.txt')
            # word embedding
            self.text = np.array(
                [np.concatenate([self.glove.get(w, np.zeros(shape=(25,))).reshape((1, 25)) for w in s[:self.maxlength]])
                 for s in self.text])
            # pad with zero vectors
            self.text = [np.concatenate((A, np.zeros(shape=((self.maxlength - A.shape[0]), 25))), axis=0) for A in
                         self.text]
            # numpy array of shape (samples,maxlength,embeddinglength)
            self.text = np.asarray(self.text)

########################################################################################################################
class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    Example: model.add(ClusteringLayer(n_clusters=10))

    Args:
        | n_clusters (int): number of clusters.
        | weights (list): list of numpy array with shape (n_clusters, n_features) witch represents the initial cluster centers.
        | alpha (float): parameter in Student's t-distribution (default 1.0).
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        """ Define the trainable weights of the Keras layer.
        Args:
            | input_shape (tuple): tuple of shape (n_samples, n_features)
        """
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ Logic of the Kera's layer. Here use the student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.

        Args:
            | inputs (2D tensor): the variable containing data with shape=(n_samples, n_features)
        Returns:
            | q (2D tensor): student's t-distribution or soft labels for each sample with shape=(n_samples, n_clusters)

        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        """ In case your layer modifies the shape of its input, you should specify here the shape transformation logic.

        Args:
            | input_shape (tuple): tuple of shape (n_samples, n_features)
        Returns:
            | (n_samples, n_clusters): the new shape after transformation
        """
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DCEC(object):
    """
    Define the Deep Clustering Image/Text Autoencoder Model

    Dependencies:
        | ClusteringLayer(Layer): clustering layer converts input sample (feature) to soft label
        | AEM(): Defines the AutoEncoder model with no clustering present using Keras.

    Args:
        | file_names (list): list locating the pickle file youtube data files
        | total_sample_cnt (int): total number of samples (with images) in all the youtube data files
        | n_clusters (int): number of clusters.
        | weights (list): list of numpy array with shape (n_clusters, n_features) witch represents the initial cluster centers.
        | alpha (float): parameter in Student's t-distribution (default 1.0).
    """
    def __init__(self, n_clusters=10, alpha=1.0, file_names=None, folder_name=None):
        super(DCEC, self).__init__()
        self.file_names = file_names # filenames where the data is located
        self.folder_name = folder_name
        self.total_sample_cnt = 100000  # total number of samples in the datafiles (countTotalSamples())
        self.batch_size = 32 # number of samples in each batch
        self.epochs = 3 # number epochs to train the autoencoder
        self.noise_factor = 0.01 # noise factor to add during training

        self.text_maxlength = 10
        self.text_feature_dim = 25 # depends on Glove model used

        self.autoencoder_weight_file = self.folder_name+'/pretrain_aem_model.h5'
        self.n_clusters = n_clusters
        self.y_pred = []  # cluster index for each sample
        self.y_pred_last = [] # previous cluster centers for each sample
        self.alpha = alpha


        self.build_model()


    def countTotalSamples(self):
        """Counts the total number of samples from all the datafiles
        Args:
            | file_names (list): list locating the pickle file youtube data files
        Returns:
            | cnt (int): total number of samples (with images) in all the youtube data file
            """
        cnt = 0
        for file in self.file_names:
            print("Counting samples in datafile: {}".format(file))
            datalist = youtube.ytf.saveloadList([], file, 0)
            datalist = [datalist[m].image for m in range(len(datalist)) if len(datalist[m].image) != 0]
            Xdata = np.array(datalist)
            cnt = cnt + Xdata.shape[0]

        self.total_sample_cnt

    def dataGenerator(self, noise_factor, infinite):
        """Data generator for Autoencoder and Deep Embedded Clustering methods

        Args:
            | batch_size (int): number of samples in each batch
            | noise_factor (float): standard deviation of the noise to apply to image
            | file_names (list): list locating the pickle file youtube data files
            | infinite (bool): infinite generator, or only one pass over data
        Returns:
            | generator: generates samples of shape (batch_size, x_dim, y_dim, channels) from the datafiles
        """
        # infinite loop condition
        textparser = TextMLParser(maxlength=self.text_maxlength)
        file_cnt = 0
        while True:
            # Generate Batches
            # load data
            print("Gathering data from file: {}".format(self.file_names[file_cnt]))
            datalist = youtube.ytf.saveloadList([], self.file_names[file_cnt], 0)
            # only keep complete data
            datalist_img = [datalist[m].image for m in range(len(datalist)) if len(datalist[m].image) != 0]
            Xdata_img = np.array(datalist_img) #shape=(samples, 40, 80, 3)
            datalist_text = [datalist[m].title for m in range(len(datalist)) if len(datalist[m].image) != 0]
            textparser.loadtext(datalist_text)
            Xdata_text = textparser.text  # shape=(samples, sequence_length, feature_dim)
            del datalist
            # number of samples
            n_samples = Xdata_img.shape[0]
            # add Gaussian noise
            # noise_factor = 0.05
            Xdatanoise = Xdata_img + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=Xdata_img.shape)
            del Xdata_img
            Xdatanoise = np.clip(Xdatanoise, 0., 1.)
            # provide samples
            for n in range(n_samples // self.batch_size):
                yield ([Xdatanoise[n * self.batch_size:(n + 1) * self.batch_size], Xdata_text[n * self.batch_size:(n + 1) * self.batch_size]],
                [Xdatanoise[n * self.batch_size:(n + 1) * self.batch_size], Xdata_text[n * self.batch_size:(n + 1) * self.batch_size]])
            # increase file counter to process new data
            if file_cnt < len(file_names) - 1:
                file_cnt = file_cnt + 1
            elif infinite:
                file_cnt = 0
            else:
                break

    def featureGenerator(self, noise_factor, infinite):
        """Data generator for Autoencoder and Deep Embedded Clustering methods

        Args:
            | batch_size (int): number of samples in each batch
            | noise_factor (float): standard deviation of the noise to apply to image
            | file_names (list): list locating the pickle file youtube data files
            | infinite (bool): infinite generator, or only one pass over data
        Returns:
            | generator: generates samples of shape (batch_size, x_dim, y_dim, channels) from the datafiles and also includes
            the feature information defined in the 'Data' objects (lines 140-145 in youtube.ytf) in the datalist.
        """
        # infinite loop condition
        file_cnt = 0
        while True:
            # Generate Batches
            # load data
            print("Gathering data from file: {}".format(self.file_names[file_cnt]))
            datalist = youtube.ytf.saveloadList([], self.file_names[file_cnt], 0)
            # get video feature information
            viewcounts = [datalist[m].videoLabels[0] for m in range(len(datalist)) if len(datalist[m].image) != 0]
            comments = [datalist[m].videoLabels[1] for m in range(len(datalist)) if len(datalist[m].image) != 0]
            likes = [datalist[m].videoLabels[2] for m in range(len(datalist)) if len(datalist[m].image) != 0]
            dislikes = [datalist[m].videoLabels[3] for m in range(len(datalist)) if len(datalist[m].image) != 0]
            category = [datalist[m].videoNumFeatures[1] for m in range(len(datalist)) if len(datalist[m].image) != 0]
            # free up space
            del datalist
            # number of samples
            n_samples = len(viewcounts)
            # provide samples
            for n in range(n_samples // self.batch_size):
                yield (viewcounts[n * self.batch_size:(n + 1) * self.batch_size],
                       comments[n * self.batch_size:(n + 1) * self.batch_size],
                       likes[n * self.batch_size:(n + 1) * self.batch_size],
                       dislikes[n * self.batch_size:(n + 1) * self.batch_size],
                       category[n * self.batch_size:(n + 1) * self.batch_size])
            # increase file counter to process new data
            if file_cnt < len(file_names) - 1:
                file_cnt = file_cnt + 1
            elif infinite:
                file_cnt = 0
            else:
                break

    def Autoencoder(self):
        """
        Defines the AutoEncoder model with no clustering present using Keras.
        Args:
            | param1: This is the first param.
        Returns:
            | autoencoder (keras.model.Model): Kera's autoencoder model where the model inputs and outputs are defined.
        """
        # ENCODER
        input_text = Input(shape=(self.text_maxlength, self.text_feature_dim))
        x_text = LSTM(150)(input_text)
        x_text = Dense(64, activation='relu')(x_text)

        input_img = Input(shape=(40, 80, 3))  # 3ch=RGB, 40 x 80
        x_img = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)  # nb_filter, (nb_row, nb_col)
        x_img = MaxPooling2D((2, 2), padding='same')(x_img)
        x_img = Conv2D(8, (3, 3), activation='relu', padding='same')(x_img)
        x_img = MaxPooling2D((2, 2), padding='same')(x_img)
        x_img = Conv2D(8, (3, 3), activation='relu', padding='same')(x_img)
        x_img = MaxPooling2D((2, 2), padding='same')(x_img)
        x_img = Flatten()(x_img)

        x_comb = Concatenate()([x_img, x_text])
        x_comb = Dense(128, activation='relu')(x_comb)
        x_comb = Dense(64, activation='relu')(x_comb)
        encoded = Dense(200, activation='linear', name='embedding')(x_comb)

        # DECODER
        x_img = Dense(400, activation='relu')(encoded)
        x_img = Reshape((5, 10, 8))(x_img)
        x_img = Conv2D(8, (3, 3), activation='relu', padding='same')(x_img)
        x_img = UpSampling2D((2, 2))(x_img)
        x_img = Conv2D(8, (3, 3), activation='relu', padding='same')(x_img)
        x_img = UpSampling2D((2, 2))(x_img)
        x_img = Conv2D(16, (3, 3), activation='relu', padding='same')(x_img)
        x_img = UpSampling2D((2, 2))(x_img)
        decoded_img = Conv2D(3, (5, 5), activation='sigmoid', padding='same')(x_img)

        x_text = RepeatVector(self.text_maxlength)(encoded)
        x_text = LSTM(150, return_sequences=True)(x_text)
        decoded_text = TimeDistributed(Dense(self.text_feature_dim, activation='sigmoid'))(x_text)

        # construct autoencoder
        autoencoder = Model([input_img, input_text], [decoded_img, decoded_text])
        return autoencoder

    def build_model(self):
        #AUTOENCODER
        self.aem = self.Autoencoder()
        self.aem.compile(loss=['mse', 'mse'], optimizer='rmsprop', metrics=['acc', 'acc'])
        #self.aem.compile(loss='mse', optimizer='rmsprop', metrics='acc')
        #ENCODER
        hidden = self.aem.get_layer(name='embedding').output
        self.encoder = Model(inputs=self.aem.input, outputs=hidden)
        #DEEPEMBEDDEDCLUSTERING
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        self.model = Model(inputs=self.aem.input, outputs=[clustering_layer, *self.aem.output])
        self.model.compile(loss=['kld', 'mse', 'mse'], loss_weights=[1, 1, 1], optimizer='rmsprop')

    def init_autoencoder(self, gen_obj):
        '''Pretrains the AutoEncoder weights prior to Clustering'''
        if os.path.isfile(self.autoencoder_weight_file):
            print('Loaded previously trained autoencoder weights.')
            self.aem.load_weights(self.folder_name+'/pretrain_aem_model.h5')
        else:
            print('...Pretraining Autoencoder...')
            self.aem.fit_generator(generator=gen_obj,
                                   steps_per_epoch = self.total_sample_cnt//self.batch_size,
                                   epochs=self.epochs,
                                   verbose=0,
                                   callbacks=[])
            self.aem.save(self.autoencoder_weight_file)
            print('Pretrained weights are saved to '+self.folder_name+'/pretrain_aem_model.h5')
            print('Complete')

    def init_clusters(self, gen_obj_finite):
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        Z = np.concatenate([self.encoder.predict(k[0]) for k in gen_obj_finite])  # construct encoded features
        self.y_pred = kmeans.fit_predict(Z) # cluster index for each sample
        self.y_pred_last = np.copy(self.y_pred) # previous cluster centers for each sample
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_]) # set the center of each cluster
        print('Complete cluster initialization')
        
    def load_dcec_weights(self, dcec_weights_file):
        self.model.load_weights(dcec_weights_file)

    def extract_feature(self, X):  # extract embedding
        '''given input X return the embedding'''
        return self.encoder.predict(X)
    
    def predict(self, X):
        '''given input X return the cluster association'''
        q, _, _ = self.model.predict(X, verbose=0)
        return q.argmax(1)

    def get_latent_cluster(self):
        print("Constructing Latent Cluster information ...")
        gen_obj_finite = self.dataGenerator(noise_factor=0.0, infinite=False)
        latent_rep = []
        cluster_labels = []
        for k in gen_obj_finite:
            latent_rep.extend(self.extract_feature(k[0]))
            cluster_labels.extend(self.predict(k[0]))
        print("Gathering feature information ...")
        gen_feature_obj = self.featureGenerator(noise_factor=0.0, infinite=False)
        viewcounts = []
        comments = []
        likes = []
        dislikes = []
        category = []
        for obj in gen_feature_obj:
            viewcounts.extend(obj[0])
            comments.extend(obj[1])
            likes.extend(obj[2])
            dislikes.extend(obj[3])
            category.extend(obj[4])
        return latent_rep, cluster_labels, viewcounts, comments, likes, dislikes, category

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def fit(self, maxiter=10, tol=1e-3, update_interval=140):
        '''fit DCEC model to data'''

        # Step 1: pretrain autoencoder
        gen_obj = self.dataGenerator(noise_factor=0.01, infinite=True)
        self.init_autoencoder(gen_obj)

        # Step 2: initialize cluster centers using k-means
        gen_obj_finite = self.dataGenerator(noise_factor=0.0, infinite=False)
        self.init_clusters(gen_obj_finite)

        # Step 3: deep clustering
        index = 0
        gen_obj = self.dataGenerator(noise_factor=0.01, infinite=False)
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                # save model weights
                self.model.save(self.folder_name+'/dcec_model_{}.h5'.format(ite))
                # update the auxiliary target distribution p
                print('Updating auxiliary target distribution p')
                gen_obj_finite = self.dataGenerator(noise_factor=0.0, infinite=False)
                q = np.concatenate([self.model.predict(k[0], verbose=0)[0] for k in gen_obj_finite])
                p = self.target_distribution(q)

                # compute max batch iterations
                max_batch_iteration = p.shape[0]//self.batch_size-1
                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                self.y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reached tolerance threshold. Stopping training.")
                    break
                
            # train on batch
            Xtrain_batch = next(gen_obj)[0] #make this finite as well
            loss = self.model.train_on_batch(x=Xtrain_batch, y=[p[index * self.batch_size:(index + 1) * self.batch_size], *Xtrain_batch])
            if index % 100 == 0:
                print('Iteration: {}, Loss: {} and index: {}'.format(ite,loss,index))
            if index == max_batch_iteration:
                index = 0
                gen_obj = self.dataGenerator(noise_factor=0.01, infinite=False)
            else:
                index += 1
        print("Minimum Loss on frame: ",loss)        

########################################################################################################################
file_names = ['./Datalist/datalist1_0_10_rgb.pkl',
              './Datalist/datalist2_11_20_rgb.pkl',
              './Datalist/datalist3_21_26_rgb.pkl',
              './Datalist/datalist4_27_35_rgb.pkl',
              './Datalist/datalist5_36_43_rgb.pkl',
              './Datalist/datalist6_44_77_rgb.pkl',
              './Datalist/datalist7_rgb.pkl',
              './Datalist/datalist8_rgb.pkl',
              './Datalist/datalist9_rgb.pkl',
              './Datalist/datalist10_rgb.pkl',
              './Datalist/datalist11_rgb.pkl',
              './Datalist/datalist12_rgb.pkl',
              './Datalist/datalist13_rgb.pkl',
              './Datalist/datalist14_rgb.pkl']

########################################################################################################################
# Deep Embedded Clustering AutoEncoder
for n_clust in [2,3,4,5,6,7,8,9,10,11]:
     # create temp folder
     os.mkdir('./temp_n{}_d200_e3'.format(n_clust))
     dcec = DCEC(n_clusters=n_clust, alpha=1.0, file_names=file_names, folder_name='./temp_n{}_d200_e3'.format(n_clust))
     dcec.fit(maxiter=11000, tol=3e-3, update_interval=1000) #made change to tolerance level here

########################################################################################################################
# t-distributed Stochastic Neighbor Embedding (tSNE) 2D representation of Video/Text Embeddings
#import scipy.io as sio
#import pandas as pd
#from MulticoreTSNE import MulticoreTSNE as TSNE

# for n_clust in [2,3,4,5,6,7,8,9,10,11]:
#     dcec = DCEC(n_clusters=n_clust, alpha=1.0, file_names=file_names, folder_name='./temp_n{}_d200_e3'.format(n_clust))
#     dcec.model.load_weights('./temp_n{}_d200_e3/dcec_model_0.h5'.format(n_clust))
#     latent_rep, cluster_labels, viewcounts, comments, likes, dislikes, category = dcec.get_latent_cluster()

#     # save data to pickle file format
#     with open('./deep_embeddings_0/deep_embedding_n{}_d200_e3.pkl'.format(n_clust), 'wb') as file:
#         pickle.dump([latent_rep, cluster_labels, viewcounts, comments, likes, dislikes, category], file)

#     # save data to matlab file format
#     sio.savemat('./deep_embeddings_0/deep_embedding_n{}_d200_e3.mat'.format(n_clust), {'frame': cluster_labels, 'viewcount': viewcounts, 'comments': comments, 'likes': likes, 'dislikes': dislikes, 'category': category})

#     # save data to csv file
#     df = pd.DataFrame(data={'frame': cluster_labels, 'viewcount': viewcounts, 'comments': comments, 'likes': likes, 'dislikes': dislikes, 'category': category})
#     df.to_csv(path_or_buf='./deep_embeddings_0/deep_embedding_n{}_d200_e3.csv'.format(n_clust), index=False)

# for n_clust in [2,3,4,5,6,7,8,9,10,11]:
#     print("Processing cluster: {} ....".format(n_clust))
#     with open('./deep_embeddings_0/deep_embedding_n{}_d200_e3.pkl'.format(n_clust), 'rb') as file:  # Python 3: open(..., 'rb')
#         latent_rep, cluster_labels, viewcounts, comments, likes, dislikes, category = pickle.load(file)

#     tsne = TSNE(n_jobs=4, perplexity=30.0, verbose=5)
#     latent_rep = np.array(latent_rep)
#     viz_latent_rep = tsne.fit_transform(latent_rep)

#     # save data to pickle file format
#     with open('./deep_embeddings_0/deep_latent_n{}_d200_e3.pkl'.format(n_clust), 'wb') as file:
#         pickle.dump([viz_latent_rep, cluster_labels, viewcounts, comments, likes, dislikes, category], file)

########################################################################################################################
