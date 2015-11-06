import cudarray as ca
import numpy as np
from ..input import Input


class TransferInput(Input):
    def __init__(self, x1, x2, batch_size=128):
        super(TransferInput, self).__init__(x1, batch_size)
        self.x2 = x2
        self.batch_size2 = batch_size if batch_size > 0 else x2.shape[0]
        self.n_samples2 = x2.shape[0]
        self.n_batches2 = int(np.ceil(float(self.n_samples2) / self.batch_size))

    def batches(self, phase = 'train', domain = 'target'):
        if phase == 'train':
            for batch_start, batch_stop in self._batch_slices2(domain = 'target'):
                x1_batch = ca.array(self.x[batch_start:batch_stop])
                x2_batch = ca.array(self.x2[batch_start:batch_stop])            
                yield {'x1': x1_batch, 'x2': x2_batch}
        elif phase == 'test':
            if domain == 'target':        
                for batch_start, batch_stop in self._batch_slices2(domain):
                    x1_batch = ca.array(self.x[batch_start:batch_stop])
                    yield {'x1': x1_batch}
            elif domain == 'source':
                for batch_start, batch_stop in self._batch_slices2(domain):
                    x2_batch = ca.array(self.x2[batch_start:batch_stop])            
                    yield {'x2': x2_batch}
            
    def _batch_slices2(self, domain = 'target'):
        if domain == 'target':        
            for b in range(self.n_batches):
                batch_start = b * self.batch_size
                batch_stop = min(self.n_samples, batch_start + self.batch_size)
                yield batch_start, batch_stop
        elif domain == 'source':
            for b in range(self.n_batches2):
                batch_start = b * self.batch_size2
                batch_stop = min(self.n_samples2, batch_start + self.batch_size2)
                yield batch_start, batch_stop
            
                      
                      
    @property
    def x_shape(self):
        return (self.batch_size,) +  self.x.shape[1:] + self.x2.shape[1:] 

    @property
    def shapes(self):
        return {'x_shape': self.x_shape}



class SupervisedTransferInput(TransferInput):
    def __init__(self, x1, x2, y, batch_size=128):
        super(SupervisedTransferInput, self).__init__(x1, x2, batch_size)
        if x1.shape[0] != y.shape[0]:
            raise ValueError('shape mismatch between x and y')
        self.y = y

    def batches(self):
        for batch_start, batch_stop in self._batch_slices():
            x1_batch = ca.array(self.x[batch_start:batch_stop])
            x2_batch = ca.array(self.x2[batch_start:batch_stop])
            y_batch = ca.array(self.y[batch_start:batch_stop])
            yield {'x1': x1_batch, 'x2': x2_batch, 'y': y_batch}

    @property
    def y_shape(self):
        return (self.batch_size,) + self.y.shape[1:]
    
    @property
    def shapes(self):
        return {'x_shape': self.x_shape, 'y_shape': self.y_shape}