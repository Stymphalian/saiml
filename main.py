import os
import timeit
import time
import math
from typing import *
from pprint import pprint

import autograd2 as ag
from devices import xp
from layers import *
from dataloader.mnist import MnistDataloader
import cupy as cp
import numpy as np
from devices import xp
from matplotlib import pyplot as plt
import optimizer
import trainer
import utils

def preprocess_data(x, y, limit=None):
    # reshape and normalize input data
    x = xp.array(x)
    # x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.reshape(x.shape[0], 1, 28, 28)
    x = x.astype("float64") / 255     # between 0 and 1.0
    x = 2*x - 1

    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = xp.array(y)
    if limit:
        return x[:limit], y[:limit]
    else:
        return x, y
    
class Noiser(Module):
    def __init__(
        self,
        steps=100,
        beta_start=1e-4,
        beta=0.8,
    ):
        super().__init__()
        self.steps = steps
        self.beta_start = beta_start
        self.beta = beta

        self.betas = xp.linspace(self.beta_start, self.beta, self.steps) # (s)
        self.alphas = 1.0 - self.betas                                   # (s)
        cum_alphas = xp.cumprod(self.alphas, axis=0)                     # (s)
        self.means = xp.sqrt(cum_alphas)                                 # (s)
        vars = 1 - cum_alphas                                            # (s)
        self.stds = xp.sqrt(vars)                                        # (s)

        # Make a "before" padded version of the variances.
        # This allows us to do indexing for t-1 when t == 0 and not get
        # index out of bounds error and keep our vectorized implementation
        self.vars = xp.pad(vars, (1, 0), constant_values=0)              # (s)

    def get(self, x, val, t):
        y = val[t.value()]                              # (...b, 1)
        y = y[..., xp.newaxis, xp.newaxis]              # (...b, 1, 1, 1)
        y = xp.broadcast_to(y, x.shape)                 # (...b, ch, xh, xw)
        return ag.Tensor(y)

    def forward(self, x, t):
        assert x.ndim >= 3, "x0.shape must be atleast (...b, ch, xh, xw)"
        assert t.ndim >= 1, "t.shape must be atleast (...b, 1)"
        assert x.shape[:-3] == t.shape[:-1]
        assert t.shape[-1] == 1

        noise = ag.Tensor(xp.random.normal(size=x.shape))           # (...b, ch, xh, xw)
        mean = self.get(x, self.means, t)                           # (...b, ch, xh, xw)
        std = self.get(x, self.stds, t)                             # (...b, ch, xh, xw)

        y = mean*x + std*noise                                      # (...b, ch, xh, xw)
        return y, noise

class ResidualFeedForward(Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.linear = Linear(in_shape, out_shape)
        self.norm = LayerNorm2((out_shape,))
        self.relu = ReLU()
        self.params = [self.linear, self.norm, self.relu]

    def forward(self, x):
        y = self.linear.forward(x)      # (b, o)
        y = self.relu.forward(y)        # (b, o)
        y = self.norm.forward(y)        # (b, o)
        return y

class PositionalEmbedding(Module):
    def __init__(self, seq_len, embed_size):
        super().__init__()
        self.seq_len = seq_len
        self.embed_size = embed_size

        w = np_normal((seq_len, embed_size))
        for pos in range(seq_len):
            for k in range(embed_size):
                if k%2 == 0:
                    z1 = xp.sin(pos / xp.power(10000, (2*k) // self.embed_size))
                    w[pos][k] = z1 
                else:
                    z2 = xp.cos(pos / xp.power(10000, (2*k) // self.embed_size))
                    w[pos][k] = z2
        self.w = ag.Tensor(w, requires_grad=True)   # (n, k)
        self.w.set_name("PositionalEncoding.w")
        self.eye = xp.eye(self.seq_len)

        self.params = [self.w]

    def onehot(self, x):
        # b, t = x.shape
        x = self.eye[x.value()]  # (b, t, n)
        x = ag.Tensor(x)         # (b, t, n)
        return x

    def forward(self, x):
        # b, t = x.shape
        z = self.onehot(x)             # (b, t, n)
        z = ag.batch_matmul(z, self.w) # (b,t,n)x(n,k) => (b,t,k)
        return z

class Block(Module):
    def __init__(
            self, 
            channels_in, 
            channels_out, 
            time_dims,
            kernel_size=3,
            downsample=True,
        ):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.time_dims = time_dims
        self.kernel_size = kernel_size
        self.downsample = downsample

        self.time_proj = Linear(time_dims, channels_out)
        self.time_relu = ReLU()
        self.time_norm = LayerNorm2((channels_out,))

        if downsample:
            self.conv1 = Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=1, padding=1)
            self.final = Conv2d(channels_out, channels_out, kernel_size=4, stride=2, padding=1)
        else:
            # the 2*channels_in is to handle the residual connection which was concatenated
            # into the channels of the input.
            # self.conv1 = Conv2d(2 * channels_in, channels_out, kernel_size=kernel_size, stride=1,padding=1)
            self.conv1 = Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=1, padding=1)
            self.final = Conv2dTranspose(channels_out, channels_out, kernel_size=4, stride=2, padding=1)
        self.conv1_relu = ReLU()
        self.conv1_norm = LayerNorm2((channels_out,1,1))

        self.conv2 = Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)
        self.conv2_relu = ReLU()
        self.conv2_norm = LayerNorm2((channels_out,1,1))

        self.params = [
            self.time_proj, self.time_relu, self.time_norm, 
            self.conv1, self.conv1_relu, self.conv1_norm,
            self.conv2, self.conv2_relu, self.conv2_norm,
            self.final
        ]
        
    def forward(self, x, t):
        # b, ch, xh, xw = x.shape
        # b, td = t.shape
        t = self.time_proj(t)        # (b, o)
        t = self.time_relu(t)        # (b, o)
        t = self.time_norm(t)        # (b, o)
        t = ag.reshape(t, (t.shape + (1,1)))   # (b, o, 1, 1)

        z = self.conv1(x)            # (b, k, yh, yw)
        z = self.conv1_relu(z)       # (b, k, yh, yw)
        z = self.conv1_norm(z)       # (b, k, yh, yw)

        zt = z + t         
        zt = self.conv2(zt)          # (b, k, yh, yw)
        zt = self.conv2_relu(zt)     # (b, k, yh, yw)
        zt = self.conv2_norm(zt)     # (b, k, yh, yw)

        zt = self.final(zt)          # (b, k, yh, yw)
        return zt        


class DeNoiser(Module):
    def __init__(
            self,
            input_shape=(1,28,28),
            time_embed_dims=32,
            channel_seq=(16,32,64,128),
        ):
        super().__init__()
        self.input_shape = input_shape,
        self.input_channels = input_shape[-3]
        self.input_height = input_shape[-2]
        self.input_width = input_shape[-1]
        self.time_embed_dims = time_embed_dims
        self.channel_seq = channel_seq
        self.reverse_channel_seq = list(reversed(self.channel_seq))

        input_size = self.input_channels * self.input_height * self.input_width

        self.flatten = Reshape((-1, input_size))
        self.positional_embedding = Sequence([
            PositionalEmbedding(input_size, self.time_embed_dims),
            Reshape((-1, self.time_embed_dims)),
        ])
        self.downsampling = [
            Block(ci, co, self.time_embed_dims)
            for (ci, co) in zip(self.channel_seq[:-1], self.channel_seq[1:])
        ]
        self.upsampling = [
            Block(ci, co, self.time_embed_dims, downsample=False)
            for (ci, co) in zip(self.reverse_channel_seq[:-1], self.reverse_channel_seq[1:])
        ]
        print([(ci,co) for (ci, co) in zip(self.channel_seq[:-1], self.channel_seq[1:])])
        print([(ci, co) for (ci, co) in zip(self.reverse_channel_seq[:-1], self.reverse_channel_seq[1:])])
        self.conv1 = Conv2d(self.input_channels, self.channel_seq[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(self.channel_seq[0], self.input_channels, kernel_size=1, stride=1, padding=0)

        self.params = [
            self.conv1,
            self.conv2,
            self.positional_embedding,
        ] + self.downsampling + self.upsampling

    def forward(self, x, t):
        t = self.positional_embedding(t)         # (b, td)
        
        x = self.conv1(x)                        # (b, k, yh, yw)
        for ds in self.downsampling:
            x = ds(x, t)
        for us in self.upsampling:   
            x = us(x, t)
        x = self.conv2(x)                         # (b, k, yh, yw)
        return x
    


class Diffusion(Module):
    def __init__(
            self,
            input_shape=(1,28,28),
            time_embed_dims=32,
            steps=100,
            beta_start=1e-4,
            beta=0.3):
        super().__init__()
        self.input_shape = input_shape
        self.time_embed_dims = time_embed_dims
        self.steps = steps
        self.beta_start = beta_start
        self.beta = beta
        
        self.noiser = Noiser(steps, beta_start, beta)
        self.denoiser = DeNoiser(input_shape, time_embed_dims)
        self.params = [self.noiser, self.denoiser]

    def forward(self, x0, t):
        assert x0.ndim >= 3, "x0.shape must be atleast (...b, ch, xh, xw)"
        assert t.ndim >= 1, "t.shape must be atleast (...b, 1)"
        assert x0.shape[:-3] == t.shape[:-1], "x0 and t must be of the same batch dimensions"
        noisy_image, true_noise = self.noiser.forward(x0, t)        # (...b, ch, xh, xw)
        pred_noise = self.denoiser.forward(noisy_image, t)          # (...b, ch, xh, xw)
        return pred_noise, true_noise
        # return noisy_image, noise_pred

    def loss(self, pred_noise, true_noise):
        loss = ag.l2_loss(pred_noise, true_noise)
        return loss
    
    def sample(self, x, t):
        assert x.ndim >= 3, "x.shape must be atleast (...b, ch, xh, xw)"
        assert t.ndim >= 1, "t.shape must be atleast (...b, 1)"
        assert x.shape[:-3] == t.shape[:-1], "x0 and t must be of the same batch dimensions"

        pred_noise, _ = self.forward(x, t)              # (...b, ch, xh, xw)
        
        # Get the predicted mean
        # (1/at) * xt - (Bt / sqrt(1-std))*e'
        a = 1.0 / self.noiser.alphas                    # (num_steps,)
        b = self.noiser.betas                           # (num_steps,)
        c = 1.0 / self.noiser.stds                      # (num_steps,)
        a = self.noiser.get(x, a, t)                    # (...b, ch, xh, xw)
        b = self.noiser.get(x, b, t)                    # (...b, ch, xh, xw)
        c = self.noiser.get(x, c, t)                    # (...b, ch, xh, xw)
        pred_mean = a * (x - (b/c)*pred_noise)          # (...b, ch, xh, xw)

        # Get the posterior variance
        # q(x_t-1 | (xt,x0)) = Bt * (1 - at) / (1 - vars)
        a = self.noiser.get(x, 1.0 - self.noiser.alphas, t) # (...b, ch, xh, xw)
        b = self.noiser.get(x, self.noiser.betas, t)        # (...b, ch, xh, xw)
        c = self.noiser.get(x, 1.0 - self.noiser.vars, t-1) # (...b, ch, xh, xw)
        std = ag.sqrt(b * (a/c))                            # (...b, ch, xh, xw)
        
        noise = ag.Tensor(xp.random.normal(size=x.shape))   # (...b, ch, xh, xw)
        x1 = pred_mean + std*noise                          # (...b, ch, xh, xw)

        return x1
    
    def generate(self, batch_size):
        x = xp.random.normal(size=(batch_size,) + self.input_shape)
        x = ag.Tensor(x)
        for step in range(self.steps, -1, -1):
            t = xp.full((batch_size, 1), step)
            x = self.sample(x, ag.Tensor(t))
        return x

def plot_images(imgs):
    imgs = imgs.value().get()
    num_rows = 5
    num_cols = 5
    plt.figure(figsize=(12,12))
    for i in range(num_rows*num_cols):
        if i >= imgs.shape[0]:
            break
        y = imgs[i].reshape(28, 28)
        y = (y + 1) / 2
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(y, cmap='gray')
    plt.show(block=True)

def plot_noise_distribution(noise, predicted_noise):
    plt.hist(noise.numpy().flatten(), density = True, alpha = 0.8, label = "ground truth noise")
    plt.hist(predicted_noise.numpy().flatten(), density = True, alpha = 0.8, label = "predicted noise")
    plt.legend()
    plt.show(block=True)

def plot_noise_prediction(noise, predicted_noise):
    # plt.figure(figsize=(15,15))
    f, ax = plt.subplots(1, 2, figsize = (5,5))
    noise = noise.value().get()
    predicted_noise = predicted_noise.value().get()
    noise = (noise + 1) / 2
    predicted_noise = (predicted_noise + 1) / 2
    ax[0].imshow(noise.reshape(28,28))
    ax[0].set_title(f"ground truth noise", fontsize = 10)
    ax[1].imshow(predicted_noise.reshape(28,28))
    ax[1].set_title(f"predicted noise", fontsize = 10)
    plt.show(block=True)

def train(
        model: Diffusion,
        batcher,
        num_iters=100,
        minibatch_size=10,
        learning_rate=5e-4):
    context = {"optimizer": optimizer.RMSProp(lr=learning_rate)}
    print("Learning Rate: {}".format(context["optimizer"].learning_rate))

    for iter in range(num_iters):
        start = timeit.default_timer()

        # setup the data
        xs,_ = batcher.sample(minibatch_size)
        ts = ag.Tensor(xp.random.randint(0, model.steps, (minibatch_size, 1)))
        idx = xp.random.choice(xs.shape[0], xs.shape[0], replace=False)
        xs = xs[idx]
        ts = ts[idx]

        pred, noise = model.forward(xs, ts)
        loss = model.loss(pred, noise)
        loss.backward()
        model.backward(context)
        
        end = timeit.default_timer()
        loss = float(loss.value())
        time_taken = end - start
        if (num_iters//10) > 0 and iter % (num_iters//10) == 0:
            print(f"[{iter}] Minibatch Loss: {loss:.6f} - time taken: {time_taken:.4f}")
            timestr = time.strftime("%Y%m%d")
            model.checkpoint(f"checkpoints/checkpoint_{timestr}.npy")


    plot_noise_distribution(noise, pred)             
    plot_noise_prediction(noise[0], pred[0])
    timestr = time.strftime("%Y%m%d")
    model.checkpoint(f"checkpoints/checkpoint_{timestr}.npy")


def test_sequence(x_shape, seq):
    current_shape = x_shape
    for (ks, stride, pad, downsample) in seq:
        xh, xw = current_shape
        if downsample:
            nh, nw = utils.get_conv2d_height_width(
                (1, ) + current_shape,
                (1, ks, ks),
                stride, 
                pad
            )
        else:
            nh, nw = utils.get_conv2d_transpose_height_width(
                (1, ) + current_shape,
                (ks, ks),
                stride, 
                pad
            )
        current_shape = (nh, nw)
        print(f"(1, {xh}, {xw}) -> (1, {nh}, {nw})")

#  test_sequence((28,28), [
#     #conv1 
#     (3, 1, 1, True),
#     #down
#     (3, 1, 1, True),
#     (3, 1, 1, True),
#     (4, 2, 1, True),
#     #down
#     (3, 1, 1, True),
#     (3, 1, 1, True),
#     (4, 2, 1, True),
#     #down
#     (3, 1, 1, True),
#     (3, 1, 1, True),
#     (4, 2, 1, True),
#     #up
#     (3, 1, 1, True),
#     (3, 1, 1, True),
#     (4, 2, 1, False),
#     #up
#     (3, 1, 1, True),
#     (3, 1, 1, True),
#     (4, 2, 1, False),
#     #up
#     (3, 1, 1, True),
#     (3, 1, 1, True),
#     (4, 2, 1, False),
#     #conv2 
#     (1, 1, 0, True),
# ])

def main():
    np.random.seed(1337)
    cp.random.seed(1337)
    xp.random.seed(1337)

    input_path = 'data'
    training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    model = Diffusion(
        input_shape=(1, 28, 28),
        time_embed_dims=32,
        steps=300,
        beta_start=1e-4,
        beta=0.02
    )

    # model.load_checkpoint("checkpoints/checkpoint_20250422.npy")
    train(
        model,
        trainer.BatchLoader().from_arrays(x_train, y_train, num_batches=10),
        num_iters=5000,
        minibatch_size=256,
        learning_rate=1e-4
    )

    # model.load_checkpoint("checkpoints/checkpoint_20250422.npy")
    # pred, noise = model.forward(
    #     ag.Tensor(x_train[:1]),
    #     ag.Tensor(xp.array([20], dtype=xp.int32).reshape(1, 1)),
    # )
    # loss = model.loss(pred, noise)
    # print(loss)
    # plot_noise_prediction(noise[0], pred[0])
    # plot_noise_distribution(noise[:1], pred[:1]) 

    y, noise = model.noiser.forward(
        ag.Tensor(xp.array([x_train[:1][0]]*10)),
        ag.Tensor(xp.arange(0, 300, 30, dtype=xp.int32).reshape(10, 1))
    )
    # y = model.generate(10)
    plot_images(y)
    plot_images(noise)


    # x = ag.Tensor(x_train[:10])
    # t = ag.Tensor(xp.arange(0, 200, 20, dtype=xp.int32).reshape(10, 1))
    # pred, noise = model.forward(x, t)
    # loss = model.loss(pred, noise)
    # loss.backward()
    # context = {"optimizer": optimizer.RMSProp(lr=1e-4)}
    # model.backward(context)
    # print(loss)


    
    # ys = model.noiser(x, t)
    # plt.figure(figsize=(15,15))
    # for i in range(10):
    #     y = ys.value()[i].get().reshape(28, 28)
    #     y = (y + 1) / 2
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(y, cmap='gray')
    # plt.show(block=True)

    # denoiser = DeNoiser((1, 28, 28), 32)
    # y = denoiser.forward(ag.Tensor(x_train[:1]), ag.Tensor([[5]]))

    
    # plt.axis('off')
    # num_images = 10
    # img_index = 0
    # betas_to_try = np.linspace(0.0, 1.0, 10)
    # num_rows = len(betas_to_try)
    # # for beta in betas_to_try:
    # noiser = Noiser(steps=200, beta=0.1)
    # step_size = noiser.steps // num_images
    # for step in range(0, noiser.steps, step_size):
    #     print(f"step = {step}")
    #     y = noiser.forward(ag.Tensor(x_train[:1]), ag.Tensor([[step]]))
    #     y = y.value()[0].get().reshape(28, 28)
    #     y = (y + 1) / 2
    #     plt.subplot(num_rows, num_images, img_index + 1)
    #     plt.imshow(y, cmap='gray')
    #     img_index += 1
    # plt.show(block=True)
    
    # fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    # for i, ax in enumerate(axes.flatten()):
    #     ax.imshow(y[i].value().get().reshape(28,28), cmap='gray')
    # plt.show(block=True)

    # # model = GANMnist(is_training=True)
    # image_size=784
    # latent_size=64

    # model = GANMnist(latent_size, image_size, is_training=True)
    # # model.load_checkpoint("checkpoints/checkpoint_20250418.npy")
    # train_gan(
    #     model,
    #     trainer.BatchLoader().from_arrays(x_train, y_train, num_batches=10),
    #     num_iters=100,
    #     minibatch_size=10,
    #     learning_rate=1e-4
    # )
    # # model.load_checkpoint("checkpoints/checkpoint_20250420.npy")
    # y = model.generate(10)

    # fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    # for i, ax in enumerate(axes.flatten()):
    #     ax.imshow(y[i].value().get().reshape(28,28), cmap='gray')
    # plt.show(block=True)
   

if __name__ == "__main__":
    main()
    