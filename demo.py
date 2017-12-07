import numpy as np
import chainer

import MyCRFLayer

n_label = 2

a = np.random.uniform(-1, 1, n_label).astype('f')
b = np.random.uniform(-1, 1, n_label).astype('f')

x1 = np.stack([b, a])
x2 = np.stack([a])

xs = [x1, x2]

ys = [np.random.randint(n_label,size = x.shape[0],dtype='i') for x in xs]

my_crf = MyCRFLayer.My_CRF(n_label)

loss = my_crf(xs,ys)

print('Ground Truth:')
for i,y in enumerate(ys):
    print('\tsentence {0}: [{1}]'.format(str(i),' '.join([str(label) for label in y])))

from chainer import optimizers
optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(my_crf)
optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))

print('Predictions:')
for epoch_i in range(201):
    with chainer.using_config('train', True):
        loss = my_crf(xs,ys)

        # update parameters
        optimizer.target.zerograds()
        loss.backward()
        optimizer.update()

    with chainer.using_config('train', False):
        if epoch_i % 50 == 0:
            print('\tEpoch {0}: (loss={1})'.format(str(epoch_i),str(loss.data)))
            for i, prediction in enumerate(my_crf.argmax(xs)):
                print('\t\tsentence {0}: [{1}]'.format(str(i), ' '.join([str(label) for label in prediction])))
