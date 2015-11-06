import numpy as np
import itertools
from ..base import Model, ParamMixin, PhaseMixin, float_
from ..input import Input


class TransferNetwork(Model, PhaseMixin):
    def __init__(self, target_layers, source_layers, loss):
        self.layers = target_layers
        self.layers2 = source_layers
        self.loss = loss
        self.bprop_until = next((idx for idx, l in enumerate(self.layers)
                                 if isinstance(l, ParamMixin)), 0)
        self.layers[self.bprop_until].bprop_to_x = False
        self.layers2[self.bprop_until].bprop_to_x = False
        self._initialized = False

    def _setup(self, x_shape, y_shape=None):
        # Setup layers sequentially
        if self._initialized:
            return
        next_shape = x_shape[0:2]
        for layer in self.layers:
            layer._setup(x_shape = next_shape)
            next_shape = layer.y_shape(next_shape)
        next_shape = x_shape[0::2]
        for layer in self.layers2:
            layer._setup(x_shape = next_shape)
            next_shape = layer.y_shape(next_shape)
        next_shape = self.loss.y_shape(next_shape)
        self._initialized = True

    @property
    def _params(self):
        #if (i == 1):
        all_params = [layer._params for layer in self.layers
                          if isinstance(layer, ParamMixin)]
        #elif (i == 2):
        #    all_params = [layer._params for layer in self.layers2
        #                  if isinstance(layer, ParamMixin)]
        # Concatenate lists in list
        return list(itertools.chain.from_iterable(all_params))

    @PhaseMixin.phase.setter
    def phase(self, phase):
        if self._phase == phase:
            return
        self._phase = phase
        for layer in self.layers + self.layers2:
            if isinstance(layer, PhaseMixin):
                layer.phase = phase

    def _update(self, x1, x2, y):
        self.phase = 'train'

        # Forward propagation
        for layer in self.layers:
            x1 = layer.fprop(x1)
        for layer in self.layers2:
            x2 = layer.fprop(x2)

        # Back propagation of partial derivatives
        grad1, grad2 = self.loss.grad(y, x1, x2)
        layers = self.layers[self.bprop_until:]
        for layer in reversed(layers[1:]):
            grad1 = layer.bprop(grad1)
        layers[0].bprop(grad1)

        layers2 = self.layers2[self.bprop_until:]
        for layer in reversed(layers2[1:]):
            grad2 = layer.bprop(grad2)
        layers2[0].bprop(grad2)

        return self.loss.loss(y, x1, x2)

    def features(self, input, phase = 'train' ,domain = 'target'):
        self.phase = phase
        input = Input.from_any(input)
        if domain == 'target':        
            next_shape = input.x.shape
            for layer in self.layers:
                next_shape = layer.y_shape(next_shape)
            feats = np.empty(next_shape)
            idx = 0
            for batch in input.batches(phase, domain):
                x_batch = batch['x1']
                x_next = x_batch
                for layer in self.layers:
                    x_next = layer.fprop(x_next)
                feats_batch = np.array(x_next)
                batch_size = x_batch.shape[0]
                feats[idx:idx+batch_size, ...] = feats_batch
                idx += batch_size
        elif domain == 'source':        
            next_shape = input.x2.shape
            for layer in self.layers2:
                next_shape = layer.y_shape(next_shape)
            feats = np.empty(next_shape)
            idx = 0
            for batch in input.batches(phase, domain):
                x_batch = batch['x2']
                x_next = x_batch
                for layer in self.layers2:
                    x_next = layer.fprop(x_next)
                feats_batch = np.array(x_next)
                batch_size = x_batch.shape[0]
                feats[idx:idx+batch_size, ...] = feats_batch
                idx += batch_size
        return feats

    def distances(self, input):
        self.phase = 'test'
        dists = np.empty((input.n_samples,), dtype=float_)
        offset = 0
        for batch in input.batches():
            x1, x2 = batch
            for layer in self.layers:
                x1 = layer.fprop(x1)
            for layer in self.layers2:
                x2 = layer.fprop(x2)
            dists_batch = self.loss.fprop(x1, x2)
            dists_batch = np.ravel(np.array(dists_batch))
            batch_size = x1.shape[0]
            dists[offset:offset+batch_size, ...] = dists_batch
            offset += batch_size
        return dists
