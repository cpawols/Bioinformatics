import os

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from melanoma.constants.constants import MODEL_DIR

MODEL_NAME = 'model_name'


class BaseNetwork(ABC):
    def __init__(self, configuration):
        self._conf = configuration

        self.loss = None
        self.optimizer = None
        self.train_op = None

        self.global_step = None
        self.summary = None

        self.saver = None

    @abstractmethod
    def load(self, session: tf.Session):
        """Loads model from last checkpoint."""
        checkpoint_path = tf.train.latest_checkpoint(self._conf[MODEL_DIR])
        meta_path = '{}.meta'.format(checkpoint_path)
        print("Loading model from last checkpoint {}".format(checkpoint_path))
        self.saver = tf.train.import_meta_graph(meta_path)
        self.saver.restore(session, checkpoint_path)

    @abstractmethod
    def build_graph(self):
        """Defines computation graph including cost function."""
        raise NotImplementedError("build_graph method must be implemented!")

    @abstractmethod
    def build_optimizer(self):
        """Defines optimizer for computation graph."""
        raise NotImplementedError("build_optimizer method must be implemented!")

    # TODO solve problem with different signatures of train methods.
    # @abstractmethod
    # def train(self, session: tf.Session, **parameters):
    #     raise NotImplementedError("train method must be implemented!")

    @abstractmethod
    def predict(self, session: tf.Session, features: np.array):
        """Make predictions."""
        raise NotImplementedError("predict_graph method must be implemented!")

    def merge_all(self):
        """Merges all summaries in graph."""
        self.summary = tf.summary.merge_all()

    def save(self, session: tf.Session):
        """Save all model."""
        checkpoint_path = os.path.join(self._conf[MODEL_NAME], '{}.ckpt'.format(self._conf[MODEL_NAME]))
        self.saver.save(session, checkpoint_path)
