import timeit
import time

import optimizer
import autograd2 as ag
from devices import xp
from layers import Module


def train(
        model: Module,
        batcher,
        number_epochs=2, 
        learning_rate=5e-4):
    # devices.print_memory("Forward:")
    # devices.print_memory("Backward:")
    # devices.print_memory("Model Backward:")
    context = {"optimizer": optimizer.RMSProp(lr=learning_rate)}

    number_batches = batcher.number_batches
    batch_blocks = number_batches // 10

    for epoch in range(number_epochs):
        avg_batch_err = 0
        avg_batch_train_time = 0
        context["optimizer"].batch_start(epoch=epoch)
        print("Learning Rate: {}".format(context["optimizer"].learning_rate))

        for batch_num, (x,y) in enumerate(batcher()):
            x, y = ag.Parameter(x), ag.Parameter(y)

            start = timeit.default_timer()
            pred = model.forward(x)
            loss = model.loss(pred, y)
            loss.backward()
            model.backward(context)
            end = timeit.default_timer()

            context["optimizer"].batch_step()
            
            time_taken = end - start
            avg_batch_err += float(loss.value())
            avg_batch_train_time += time_taken
            if batch_blocks > 0 and batch_num % batch_blocks == 0:
                print("Batch Number {:3d}: error {}, time taken: {:.4f}".format(batch_num, loss.value(), time_taken))
            
        model.checkpoint(f"checkpoints/checkpoint_{epoch}.npy")
        avg_batch_err /= number_batches
        avg_batch_train_time /= number_batches
        context["optimizer"].batch_end()
        print(f"Epoch {epoch+1}/{number_epochs} - Average Batch Error: {avg_batch_err} - Time Taken/batch {avg_batch_train_time}")

    timestr = time.strftime("%Y%m%d")
    model.checkpoint(f"checkpoints/checkpoint_{timestr}.npy")
                         

class BatchLoader:
    def __init__(self):
        self._batch_size = None
        self._num_batches = None
        self.x = None
        self.y = None
        self.x_split = None
        self.y_split = None
        self.pointer = 0

    @property
    def number_batches(self):
        return self._num_batches
    
    @property
    def batch_size(self):
        return self._batch_size

    def from_arrays(self, x_train, y_train, batch_size=None, num_batches=None):
        if batch_size is not None:
            assert num_batches is None
            self._batch_size = batch_size
            self._num_batches = x_train.shape[0] // batch_size
        else:
            assert batch_size is None
            self._num_batches = num_batches
            self._batch_size = x_train.shape[0] // num_batches

        assert x_train.shape[0] % self._batch_size == 0
        assert y_train.shape[0] % self._batch_size == 0

        num_splits = self.number_batches
        self.x = x_train
        self.y = y_train
        self.x_split = xp.vsplit(self.x, num_splits)
        if self.y.ndim == 1:
            y1 = self.y.reshape(self.y.shape[0], 1)
            self.y_split = xp.vsplit(y1, num_splits)
            self.y_split = [a.flatten() for a in self.y_split]
        else:
            self.y_split = xp.vsplit(self.y, num_splits)
        return self

    def __call__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
    
    def sample(self, num_samples):
        x_choice = xp.random.choice(self.x.shape[0], num_samples, replace=False)
        return self.x[x_choice], self.y[x_choice]

    def next(self):
        return zip(self.x_split, self.y_split)
        # count = 0
        # self.pointer = 0
        # while self.pointer < len(self.x_split):
        #     yield self.x_split[self.pointer], self.y_split[self.pointer]
        #     self.pointer += 1
        #     count += 1
        #     if self._num_batches is not None and count >= self._num_batches:
        #         break
