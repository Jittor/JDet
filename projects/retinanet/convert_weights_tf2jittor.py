
import argparse
import tensorflow as tf
import pickle as pk
import numpy as np

def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3,2,0,1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v

def read_ckpt(ckpt):
    reader = tf.train.NewCheckpointReader(ckpt)
    weights = {n: reader.get_tensor(n) for (n, _) in reader.get_variable_to_shape_map().items()}
    pyweights = {k: tr(v) for (k, v) in weights.items()}
    return pyweights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts ckpt weights to deepdish hdf5")
    parser.add_argument("infile", type=str,
                        help="Path to the ckpt.")  # ***model.ckpt-22177***
    parser.add_argument("outfile", type=str, nargs='?', default='',
                        help="Output file (inferred if missing).")
    args = parser.parse_args()
    # if args.outfile == '':
    #     args.outfile = os.path.splitext(args.infile)[0] + '.h5'
    # outdir = os.path.dirname(args.outfile)
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    weights = read_ckpt(args.infile)
    for k, v in weights.items():
        print(k, v.shape)
    pk.dump(weights, open(args.outfile, "wb"))
    # dd.io.save(args.outfile, weights)