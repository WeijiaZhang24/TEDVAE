
import argparse
import logging
import pandas as pd
import torch

import pyro
from tedvae_gpu import TEDVAE
from datasets import IHDP
import numpy as np

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)




def main(args,reptition = 1, path = "./IHDP/"):
    pyro.enable_validation(__debug__)
    # if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Generate synthetic data.
    pyro.set_rng_seed(args.seed)
    train, test, contfeats, binfeats = IHDP(path = path, reps = reptition, cuda = True)
    (x_train, t_train, y_train), true_ite_train = train
    (x_test, t_test, y_test), true_ite_test = test
    
    ym, ys = y_train.mean(), y_train.std()
    y_train = (y_train - ym) / ys

    # Train.
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    tedvae = TEDVAE(feature_dim=args.feature_dim, continuous_dim= contfeats, binary_dim = binfeats,
                  latent_dim=args.latent_dim, latent_dim_t = args.latent_dim_t, latent_dim_y = args.latent_dim_y,
                  hidden_dim=args.hidden_dim,
                  num_layers=args.num_layers,
                  num_samples=10)                                                                                                                                                                                                                                                                                                                                                           
    tedvae.fit(x_train, t_train, y_train,
              num_epochs=args.num_epochs,
              batch_size=args.batch_size,
              learning_rate=args.learning_rate,
              learning_rate_decay=args.learning_rate_decay, weight_decay=args.weight_decay)

    # Evaluate.
    est_ite = tedvae.ite(x_test, ym, ys)
    est_ite_train = tedvae.ite(x_train, ym, ys)

    pehe = np.sqrt(np.mean((true_ite_test.squeeze()-est_ite.cpu().numpy())*(true_ite_test.squeeze()-est_ite.cpu().numpy())))
    pehe_train = np.sqrt(np.mean((true_ite_train.squeeze()-est_ite_train.cpu().numpy())*(true_ite_train.squeeze()-est_ite_train.cpu().numpy())))
    print("PEHE_train = {:0.3g}".format(pehe_train))

    print("PEHE = {:0.3g}".format(pehe))
    return pehe, pehe_train

if __name__ == "__main__":
    # assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser(description="TEDVAE")
    parser.add_argument("--feature-dim", default=25, type=int)


    parser.add_argument("--latent-dim", default=20, type=int)
    parser.add_argument("--latent-dim-t", default=10, type=int)
    parser.add_argument("--latent-dim-y", default=10, type=int)
    parser.add_argument("--hidden-dim", default=500, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("-n", "--num-epochs", default=200, type=int)
    parser.add_argument("-b", "--batch-size", default=1000, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.01, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=1234567890, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

        
    # tedvae_pehe = main(args)
    tedvae_pehe = np.zeros((100,1))
    tedvae_pehe_train = np.zeros((100,1))
    path = "./IHDP_b/"
    for i in range(100):
            print("Dataset {:d}".format(i+1))
            tedvae_pehe[i,0], tedvae_pehe_train[i,0] = main(args,i+1, path)

