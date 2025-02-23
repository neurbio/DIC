import scanpy as sc
import numpy as np
from preprocess import normalize
from keras.utils import to_categorical

hiddensize = '128,64,128'
dropoutrate = '0.2'
import preprocess
def main():
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError('DCA requires TensorFlow v2+. Please follow instructions'
                          ' at https://www.tensorflow.org/install/ to install'
                          ' it.')
    import train
    input_file = 'E:\python\GRNC\data/MEScounts10426.h5ad'
    input_label = 'E:\python\GRNC\data/mse_pseudo_labels.npy'
    adata = sc.read(input_file)
    adata = normalize(adata, filter_min_counts=False, size_factors=True, normalize_input=False, logtrans_input=True)
    y = np.load(input_label)
    '''y = np.array([int(i) for i in range(243)])
    y[y > 136] = 1
    y[:138] = 0'''
    y = to_categorical(y, num_classes=4)

    print(adata)

    adata = preprocess.normalize(adata,
                                 size_factors=True,
                                 logtrans_input=True,
                                 normalize_input=True)
    from network import AE_types
    input_size = adata.n_vars
    output_size = adata.n_vars
    hidden_size = [int(x) for x in hiddensize.split(',')]
    hidden_dropout = [float(x) for x in dropoutrate.split(',')]
    if len(hidden_dropout) == 1:
        hidden_dropout = hidden_dropout[0]
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()

    net = AE_types['tpgg'](input_size=input_size,
                              output_size=output_size,
                              hidden_size=hidden_size,
                              l2_coef=0.0,
                              l1_coef=0.0,
                              l2_enc_coef=0.0,
                              l1_enc_coef=0.0,
                              ridge=0.0,
                              hidden_dropout=hidden_dropout,
                              input_dropout=0.0,
                              batchnorm=True,
                              activation='relu',
                              init='glorot_uniform',
                              debug=False,
                              file_path='D:/neurb/OneDrive/python/metaaus/output')

    net.save()
    net.build()
    train.train(adata, y, net)
    print(adata.X)
    pred = net.predict(adata, copy=True)
    print(pred.X)
    net.write(pred, file_path='D:/neurb/OneDrive/python/metaaus/output')

if __name__ == '__main__':
    main()
