import _pickle as pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import os, time, scipy.stats, random
import matplotlib.pyplot as mpl
from collections import defaultdict

#MEAN, MEDIAN, 10 QUANTILES
# is gene goverage 1 a quantile of gene coverage?
GET_DATA = False
CONTINUOUS_FEATURES = ["G1_mean", "G2_mean", "HK_mean", "GeneCoverage_0", "GeneCoverage_1", "Entropy_0", "Entropy_1", "PC1", "PC2"]
CATEGORICAL_FEATURES = ["cl1", "plate", "droplet"]

FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

TRAIN = ['Kyle_Anterior', 'Kyle_Middle',  'HumanEmbryo', 'Marrow_10x_G', 'Marrow_10x_E','Marrow_10x_B', 'Marrow_plate_M']
DEV = ['HSMM','Marrow_plate_G','Marrow_plate_B','Camargo']
TEST = ['HSC_10x','RegevIntestine', 'RegevDropseq', 'StandardProtocol', 'DirectProtocol','ChuCellType','Gottgens','GrunIntestine','Fibroblast_MyoF', 'Fibroblast_MFB']
QUESTIONABLE = ['AT2', 'EPI', "Astrocytoma"]

def loadData():
    df = pd.read_csv("CompiledTableNN_filtered_PCAUpdated.csv")
    datasets = df["Dataset"].unique()
    df["DatasetName"] = df["Dataset"]
    df.set_index(["Dataset"], inplace=True)
    df.sort_index(inplace=True)

    for feature in CONTINUOUS_FEATURES:
        FEATURES.append(feature + " mean")
        FEATURES.append(feature + " median")
        for i in range(0, 11, 1):
            FEATURES.append(feature + " " + str(i*10) + " percentile")
    
    for dataset in datasets:
        # Add dataset metadata to each of the features
        for feature in CONTINUOUS_FEATURES:
            df.loc[dataset, feature + " mean"] = np.mean(df.loc[dataset, feature]) 
            df.loc[dataset, feature + " median"] = np.median(df.loc[dataset, feature])
            for i in range(0, 11, 1):
                df.loc[dataset, feature + " " + str(i*10) + " percentile"] = np.percentile(df.loc[dataset, feature], i*10)
    # Adding indicators for sequencing types
    cl1 = {"Plate": 0.0, "Droplet": 0.0, "C1": 1.0}
    droplet = {"Plate": 0.0, "Droplet": 1.0, "C1": 0.0}
    plate = {"Plate": 1.0, "Droplet": 0.0, "C1": 0.0}
    
    df["cl1"] = df["SeqType"].map(cl1)
    df["plate"] = df["SeqType"].map(plate)
    df["droplet"] = df["SeqType"].map(droplet)
    
    df["Standardized_Order"] = 1 - (df["Standardized_Order"] - df["Standardized_Order"].min())/(df["Standardized_Order"]- df["Standardized_Order"].min()).max()
    trainDf = [(row["Standardized_Order"], [row[i] for i in FEATURES], row["DatasetName"], row["GeneCoverage_1"]) for idx, row in df.loc[TRAIN].iterrows()]
    print("Finished training")
    devDf = [(row["Standardized_Order"], [row[i] for i in FEATURES], row["DatasetName"], row["GeneCoverage_1"]) for idx, row in df.loc[DEV].iterrows()]
    print("Finished dev")
    testDf = [(row["Standardized_Order"], [row[i] for i in FEATURES], row["DatasetName"], row["GeneCoverage_1"]) for idx, row in df.loc[TEST].iterrows()]

    
    df.to_csv("um.csv")
    
    return trainDf, devDf, testDf

def _minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

class Config(object):
    n_features = 129
    n_classes = 1
    dropout = 0
    batch_size = 2000
    hidden_size = 200
    n_epochs = 100
    lr = 0.0005

class NeuralNetwork():
    
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape = [None, self.config.n_features], name = "input")
        self.labels_placeholder = tf.placeholder(tf.float32, shape = [None, 1], name = "output")
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=0):
        feed_dict = {self.input_placeholder: inputs_batch, self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_prediction_op(self):
        x = self.input_placeholder
        layer1 = tf.contrib.layers.fully_connected(x, self.config.hidden_size)
        layer2 = tf.contrib.layers.fully_connected(layer1, self.config.hidden_size)
        pred = tf.contrib.layers.fully_connected(layer2, 1)
        return pred

    def add_loss_op(self, pred):
        vx = pred - tf.reduce_mean(pred, axis = 0)
        vy = self.labels_placeholder - tf.reduce_mean(self.labels_placeholder, axis = 0)
        loss = -tf.reduce_sum(tf.multiply(vx,vy))/tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vx))), tf.sqrt(tf.reduce_sum(tf.square(vy))))
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, train_examples, dev_set):
        n_minibatches = len(TRAIN) + 1
        prog = tf.keras.utils.Progbar(target=n_minibatches)

        train_x = np.array([d[1] for d in train_examples])
        train_ground = np.array([d[0] for d in train_examples])
        datasets = np.array([d[2] for d in train_examples])
        train = TRAIN[:]
        for k, dataset in enumerate(train):
            x = []
            ground = []
            for i, name in enumerate(datasets):
                if name == dataset:
                    x.append(train_x[i])
                    ground.append(train_ground[i])
            x = np.array(x)
            ground = np.matrix(ground).T
            loss = self.train_on_batch(sess, x, ground)
            prog.update(k + 1, [("train loss", loss)], force=k + 1 == n_minibatches)
            train = np.random.permutation(train)

        print("Evaluating on dev set",)
        dev_x = np.array([d[1] for d in dev_set])
        dev_ground = np.array([d[0] for d in dev_set])
        datasets = np.array([d[2] for d in dev_set])
        gc = np.array([d[3] for d in dev_set])
        sum = 0
        for dataset in DEV:
                x = []
                ground = []
                gene = []
                for i, name in enumerate(datasets):
                    if name == dataset:
                        x.append(dev_x[i])
                        ground.append(dev_ground[i])
                        gene.append(gc[i])
                pred = self.predict_on_batch(sess, x)
                sum += scipy.stats.spearmanr(pred, ground)[0]

        loss = sum/float(len(DEV))
        print("- dev: {:.2f}".format(loss))
        print(scipy.stats.spearmanr(gc, dev_ground)[0])
        return loss

    def fit(self, sess, saver, train_examples, dev_set):
        best_dev_UAS = 0
        for epoch in range(self.config.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_UAS = self.run_epoch(sess, train_examples, dev_set)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                if saver:
                    print("New best dev UAS! Saving model in ./data/weights/parser.weights")
                    saver.save(sess, './data/weights/parser.weights')
            print()

    def predict_on_batch(self, sess, inputs_batch):
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

            
    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        
    def __init__(self, config):
        self.config = config
        self.build()


def main(debug = False):

    avg_train, avg_test = 0, 0
    avg_train_gc, avg_test_gc = 0, 0
    if GET_DATA:
        train_examples, dev_set, test_set = loadData()
        pickle.dump(train_examples, open("train", "wb"))
        pickle.dump(dev_set, open("dev", "wb"))
        pickle.dump(test_set, open("test", "wb"))
    
    train_examples = pickle.load(open("train", "rb"))
    dev_set = pickle.load(open("dev","rb"))
    test_set = pickle.load(open("test", "rb"))
        
    config = Config()
    
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    with tf.Graph().as_default() as graph:
        print("Building model...",)
        start = time.time()
        model = NeuralNetwork(config)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        print("took {:.2f} seconds\n".format(time.time() - start))
    graph.finalize()

    with tf.Session(graph=graph) as session:
        session.run(init_op)

        print(80 * "=")
        print("TRAINING")
        print(80 * "=")
        model.fit(session, saver, train_examples, dev_set)

        if not debug:
            print(80 * "=")
            print("TESTING")
            print(80 * "=")
            print("Restoring the best model weights found on the dev set")
            saver.restore(session, './data/weights/parser.weights')
            print("Final evaluation on test set")

            mpl.figure(figsize=(20,10))
            
            test_x = np.array([d[1] for d in test_set])
            test_ground = np.array([d[0] for d in test_set])
            datasets = np.array([d[2] for d in test_set])
            gc = np.array([d[3] for d in test_set])
            for dataset in TEST:
                x = []
                ground = []
                gene = []
                for i, name in enumerate(datasets):
                    if name == dataset:
                        x.append(test_x[i])
                        ground.append(test_ground[i])
                        gene.append(gc[i])
                pred = model.predict_on_batch(session, x)
                print(dataset, ":", scipy.stats.spearmanr(pred, ground)[0], "Gene Coverage:", scipy.stats.spearmanr(ground, gene)[0])
                avg_test += scipy.stats.spearmanr(pred, ground)[0]
                avg_test_gc += scipy.stats.spearmanr(ground, gene)[0]
                pred = [x[0] for x in pred]


                dictGeneCoverage, dictPred = defaultdict(list), defaultdict(list)
                for i, y in enumerate(ground):
                    dictPred[y].append(pred[i])
                    dictGeneCoverage[y].append(gene[i])
                dataGeneCoverage, dataPred = [], []
                sortedPred = sorted(dictPred.keys())
                for cat in sortedPred:
                    dataPred.append(dictPred[cat])
                    dataGeneCoverage.append(dictGeneCoverage[cat])
                
                mpl.title(dataset + " Set: Test" + "\n R_neural = " + str(scipy.stats.spearmanr(pred, ground)[0]) + "\n R_gene_coverage = " + str(scipy.stats.spearmanr(ground, gene)[0]))
                c = "blue"
                mpl.boxplot(dataPred, positions=np.array(range(len(dataPred)))+0.75, widths =0.25, notch=True, patch_artist=True,
                            boxprops=dict(facecolor=c, color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),)
                c = "red"
                mpl.boxplot(dataGeneCoverage, positions=np.array(range(len(dataPred)))+1.25,widths=0.25, notch=True, patch_artist=True, boxprops=dict(facecolor=c, color=c),  capprops=dict(color=c), whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c), medianprops=dict(color=c),)
                mpl.xticks(np.array(range(len(dataPred))) + 1, sortedPred)
                mpl.xlim(0, len(dataPred)+1)
                mpl.savefig("./withgenecoverage/" + dataset + "withGeneCoverage.png")
                mpl.clf()

                mpl.title(dataset + " Set: Test" + "\n R_neural = " + str(scipy.stats.spearmanr(pred, ground)[0]) + "\n R_gene_coverage = " + str(scipy.stats.spearmanr(ground, gene)[0]))
                c = "blue"
                mpl.boxplot(dataPred, positions=np.array(range(len(dataPred)))+0.75, widths =0.25, notch=True, patch_artist=True,
                            boxprops=dict(facecolor=c, color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),)
                mpl.xlim(0, len(dataPred)+1)
                mpl.savefig("./without/" + dataset + "without.png")
                mpl.clf()
                


                
            train_x = np.array([d[1] for d in train_examples])
            train_ground = np.array([d[0] for d in train_examples])
            datasets = np.array([d[2] for d in train_examples])
            gc = np.array([d[3] for d in train_examples])
            for dataset in TRAIN:
                x = []
                ground = []
                gene = []
                for i, name in enumerate(datasets):
                    if name == dataset:
                        x.append(train_x[i])
                        ground.append(train_ground[i])
                        gene.append(gc[i])
                pred = model.predict_on_batch(session, x)
                print(dataset, ":", scipy.stats.spearmanr(pred, ground)[0], "Gene Coverage:", scipy.stats.spearmanr(ground, gene)[0])
                pred = [x[0] for x in pred]
                avg_train += scipy.stats.spearmanr(pred, ground)[0]
                avg_train_gc += scipy.stats.spearmanr(ground, gene)[0]
                dictGeneCoverage, dictPred = defaultdict(list), defaultdict(list)
                for i, y in enumerate(ground):
                    dictPred[y].append(pred[i])
                    dictGeneCoverage[y].append(gene[i])
                dataGeneCoverage, dataPred = [], []
                sortedPred = sorted(dictPred.keys())
                for cat in sortedPred:
                    dataPred.append(dictPred[cat])
                    dataGeneCoverage.append(dictGeneCoverage[cat])
                
                mpl.title(dataset + " Set: Train" + "\n R_neural = " + str(scipy.stats.spearmanr(pred, ground)[0]) + "\n R_gene_coverage = " + str(scipy.stats.spearmanr(ground, gene)[0]))
                c = "blue"
                mpl.boxplot(dataPred, positions=np.array(range(len(dataPred)))+0.75, widths =0.25, notch=True, patch_artist=True,
                            boxprops=dict(facecolor=c, color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),)
                c = "red"
                mpl.boxplot(dataGeneCoverage, positions=np.array(range(len(dataPred)))+1.25,widths=0.25, notch=True, patch_artist=True, boxprops=dict(facecolor=c, color=c),  capprops=dict(color=c), whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c), medianprops=dict(color=c),)
                mpl.xticks(np.array(range(len(dataPred))) + 1, sortedPred)
                mpl.xlim(0, len(dataPred)+1)
                mpl.savefig("./withgenecoverage/" + dataset + "withGeneCoverage.png")
                mpl.clf()

                mpl.title(dataset + " Set: Train" + "\n R_neural = " + str(scipy.stats.spearmanr(pred, ground)[0]) + "\n R_gene_coverage = " + str(scipy.stats.spearmanr(ground, gene)[0]))
                c = "blue"
                mpl.boxplot(dataPred, positions=np.array(range(len(dataPred)))+0.75, widths =0.25, notch=True, patch_artist=True,
                            boxprops=dict(facecolor=c, color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),)
                mpl.xlim(0, len(dataPred)+1)
                mpl.savefig("./without/" + dataset + "without.png")
                mpl.clf()
            print("Avg R for test GC:", avg_test_gc/len(TEST), "Avg R for train GC:", avg_train_gc/len(TRAIN))
            print("Avg R for test:", avg_test/len(TEST), "Avg R for train:", avg_train/len(TRAIN))
            print("Done!")
            

if __name__ == "__main__":
    main()
