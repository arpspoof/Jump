import time
import numpy as np
from glob import glob

from utils.vae import VAE, VAETrainer
from utils.feature import FeatureConverter, get_mocap_frames

if __name__ == '__main__':
    from presets import preset
    preset.load_default()

    fc = FeatureConverter()

    import argparse

    parser = argparse.ArgumentParser(description='train pose vae ...')
    parser.add_argument('--rawdata', type=str, default='data/vae/txt', help='raw txt motion data folder')
    parser.add_argument('--epoch', type=int, default=80, help='num of epochs')
    parser.add_argument('--name', type=str, default='test', help='name of the experiment')
    args = parser.parse_args()

    import time
    stamp = time.strftime("%Y-%b-%d-%H%M%S", time.localtime())
    exp_id = "%s-%s" % (args.name, stamp)
    path = 'results/vae/' + exp_id
    
    import os
    if not os.path.exists(path):
        os.makedirs(path)

    data_file = '%s/data.npy' % path

    import glob
    motion_files = glob.iglob(args.rawdata + '/**/*.txt', recursive=True)

    frames = []

    for train_file in motion_files:
        print(train_file)
        cur_frames = get_mocap_frames(train_file)
        for fr in cur_frames:
            feature = fc.quat_to_pos(fr)
            frames.append(feature)
            frames.append(fc.mirror(feature))

    data = np.array(frames)
    np.random.shuffle(data)

    with open(data_file, 'wb') as f:
        np.save(f, data)
    
    print('%d frames processed' % len(frames))
    
    model = VAE(preset.vae.latent_dim)
    trainer = VAETrainer(model, data)

    if args.epoch is not None:
        epochs = args.epoch
        trainer.epochs = epochs

    model_file = '%s/%s.pth' % (path, args.name)
    trainer.train(model_file)
    