import argparse
import logging
import os
import numpy as np
import imageio
from torchvision.utils import make_grid
import torch
from dpsda.logging import setup_logging
from dpsda.data_loader import load_data
from dpsda.feature_extractor import extract_features
from dpsda.metrics import make_fid_stats
from dpsda.metrics import compute_fid
from dpsda.dp_counter import dp_nn_histogram
from dpsda.arg_utils import str2bool
from apis import get_api_class_from_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--api',
        type=str,
        required=True,
        choices=['DALLE', 'stable_diffusion', 'improved_diffusion'],
        help='Which foundation model API to use')
    parser.add_argument(
        '--plot_images',
        type=str2bool,
        default=True,
        help='Whether to save generated images in PNG files')
    parser.add_argument(
        '--data_checkpoint_path',
        type=str,
        default="",
        help='Path to the data checkpoint')
    parser.add_argument(
        '--data_checkpoint_step',
        type=int,
        default=-1,
        help='Iteration of the data checkpoint')
    parser.add_argument(
        '--num_samples_schedule',
        type=str,
        default='50000,'*9 + '50000',
        help='Number of samples to generate at each iteration')
    parser.add_argument(
        '--variation_degree_schedule',
        type=str,
        default='0,'*9 + '0',
        help='Variation degree at each iteration')
    parser.add_argument(
        '--num_fid_samples',
        type=int,
        default=50000,
        help='Number of generated samples to compute FID')
    parser.add_argument(
        '--num_private_samples',
        type=int,
        default=50000,
        help='Number of private samples to load')
    parser.add_argument(
        '--noise_multiplier',
        type=float,
        default=0.0,
        help='Noise multiplier for DP NN histogram')
    parser.add_argument(
        '--lookahead_degree',
        type=int,
        default=0,
        help=('Lookahead degree for computing distances between private and '
              'generated images'))
    parser.add_argument(
        '--feature_extractor',
        type=str,
        default='clip_vit_b_32',
        choices=['inception_v3', 'clip_vit_b_32', 'original'],
        help='Which image feature extractor to use')
    parser.add_argument(
        '--num_nearest_neighbor',
        type=int,
        default=1,
        help='Number of nearest neighbors to find in DP NN histogram')
    parser.add_argument(
        '--nn_mode',
        type=str,
        default='L2',
        choices=['L2', 'IP'],
        help='Which distance metric to use in DP NN histogram')
    parser.add_argument(
        '--private_image_size',
        type=int,
        default=1024,
        help='Size of private images')
    parser.add_argument(
        '--tmp_folder',
        type=str,
        default='result/tmp',
        help='Temporary folder for storing intermediate results')
    parser.add_argument(
        '--result_folder',
        type=str,
        default='result',
        help='Folder for storing results')
    parser.add_argument(
        '--data_folder',
        type=str,
        required=True,
        help='Folder that contains the private images')
    parser.add_argument(
        '--count_threshold',
        type=float,
        default=0.0,
        help='Threshold for DP NN histogram')
    parser.add_argument(
        '--compute_fid',
        type=str2bool,
        default=True,
        help='Whether to compute FID')
    parser.add_argument(
        '--fid_dataset_name',
        type=str,
        default='customized_dataset',
        help=('Name of the dataset for computing FID against. If '
              'fid_dataset_name and fid_dataset_split in combination are one '
              'of the precomputed datasets in '
              'https://github.com/GaParmar/clean-fid and make_fid_stats=False,'
              ' then the precomputed statistics will be used. Otherwise, the '
              'statistics will be computed using the private samples and saved'
              ' with fid_dataset_name and fid_dataset_split for future use.'))
    parser.add_argument(
        '--fid_dataset_split',
        type=str,
        default='train',
        help=('Split of the dataset for computing FID against. If '
              'fid_dataset_name and fid_dataset_split in combination are one '
              'of the precomputed datasets in '
              'https://github.com/GaParmar/clean-fid and make_fid_stats=False,'
              ' then the precomputed statistics will be used. Otherwise, the '
              'statistics will be computed using the private samples and saved'
              ' with fid_dataset_name and fid_dataset_split for future use.'))
    parser.add_argument(
        '--fid_model_name',
        type=str,
        default='inception_v3',
        choices=['inception_v3', 'clip_vit_b_32'],
        help='Which embedding network to use for computing FID')
    parser.add_argument(
        '--make_fid_stats',
        type=str2bool,
        default=True,
        help='Whether to compute FID stats for the private samples')
    parser.add_argument(
        '--data_loading_batch_size',
        type=int,
        default=100,
        help='Batch size for loading private samples')
    parser.add_argument(
        '--feature_extractor_batch_size',
        type=int,
        default=500,
        help='Batch size for feature extraction')
    parser.add_argument(
        '--fid_batch_size',
        type=int,
        default=500,
        help='Batch size for computing FID')
    parser.add_argument(
        '--gen_class_cond',
        type=str2bool,
        default=False,
        help='Whether to generate class labels')
    parser.add_argument(
        '--initial_prompt',
        action='append',
        type=str,
        help='Initial prompt for image generation. It can be specified '
             'multiple times to provide a list of prompts. If the API accepts '
             'prompts, the initial samples will be generated with these '
             'prompts')
    parser.add_argument(
        '--image_size',
        type=str,
        default='1024x1024',
        help='Size of generated images in the format of HxW')
    args, api_args = parser.parse_known_args()

    args.num_samples_schedule = list(map(
        int, args.num_samples_schedule.split(',')))
    variation_degree_type = (float if '.' in args.variation_degree_schedule
                             else int)
    args.variation_degree_schedule = list(map(
        variation_degree_type, args.variation_degree_schedule.split(',')))

    if len(args.num_samples_schedule) != len(args.variation_degree_schedule):
        raise ValueError('The length of num_samples_schedule and '
                         'variation_degree_schedule should be the same')

    api_class = get_api_class_from_name(args.api)
    api = api_class.from_command_line_args(api_args)

    return args, api


def log_samples(samples, additional_info, folder, plot_images):
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savez(
        os.path.join(folder, 'samples.npz'),
        samples=samples,
        additional_info=additional_info)
    if plot_images:
        for i in range(samples.shape[0]):
            imageio.imwrite(os.path.join(folder, f'{i}.png'), samples[i])


def load_samples(path):
    data = np.load(path)
    samples = data['samples']
    additional_info = data['additional_info']
    return samples, additional_info


def log_count(count, clean_count, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.savez(path, count=count, clean_count=clean_count)


def round_to_uint8(image):
    return np.around(np.clip(image, a_min=0, a_max=255)).astype(np.uint8)


def visualize(samples, packed_samples, count, folder, suffix=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    samples = samples.transpose((0, 3, 1, 2))
    packed_samples = packed_samples.transpose((0, 1, 4, 2, 3))

    ids = np.argsort(count)[::-1][:5]
    print(count[ids])
    vis_samples = []
    for i in range(len(ids)):
        vis_samples.append(samples[ids[i]])
        for j in range(packed_samples.shape[1]):
            vis_samples.append(packed_samples[ids[i]][j])
    vis_samples = np.stack(vis_samples)
    vis_samples = make_grid(
        torch.Tensor(vis_samples),
        nrow=packed_samples.shape[1] + 1).numpy().transpose((1, 2, 0))
    vis_samples = round_to_uint8(vis_samples)
    imageio.imsave(
        os.path.join(folder, f'visualize_top_{suffix}.png'), vis_samples)

    ids = np.argsort(count)[:5]
    print(count[ids])
    vis_samples = []
    for i in range(len(ids)):
        vis_samples.append(samples[ids[i]])
        for j in range(packed_samples.shape[1]):
            vis_samples.append(packed_samples[ids[i]][j])
    vis_samples = np.stack(vis_samples)
    vis_samples = make_grid(
        torch.Tensor(vis_samples),
        nrow=packed_samples.shape[1] + 1).numpy().transpose((1, 2, 0))
    vis_samples = round_to_uint8(vis_samples)
    imageio.imsave(
        os.path.join(folder, f'visualize_bottom_{suffix}.png'), vis_samples)


def log_fid(folder, fid, t):
    with open(os.path.join(folder, 'fid.csv'), 'a') as f:
        f.write(f'{t} {fid}\n')


def main():
    args, api = parse_args()
    if os.path.exists(args.result_folder):
        raise RuntimeError(f'{args.result_folder} exists')
    os.makedirs(args.result_folder)
    setup_logging(os.path.join(args.result_folder, 'log.log'))
    logging.info(f'config: {args}')
    logging.info(f'API config: {api.args}')

    all_private_samples, all_private_labels = load_data(
        data_dir=args.data_folder,
        batch_size=args.data_loading_batch_size,
        image_size=args.private_image_size,
        class_cond=args.gen_class_cond,
        num_private_samples=args.num_private_samples)

    private_classes = list(sorted(set(list(all_private_labels))))
    private_num_classes = len(private_classes)
    logging.info(f'Private_num_classes: {private_num_classes}')

    logging.info('Extracting features')
    all_private_features = extract_features(
        data=all_private_samples,
        tmp_folder=args.tmp_folder,
        model_name=args.feature_extractor,
        res=args.private_image_size,
        batch_size=args.feature_extractor_batch_size)
    logging.info(f'all_private_features.shape: {all_private_features.shape}')

    if args.make_fid_stats:
        logging.info('Computing FID stats')
        make_fid_stats(
            samples=all_private_samples,
            dataset=args.fid_dataset_name,
            dataset_res=args.private_image_size,
            dataset_split=args.fid_dataset_split,
            tmp_folder=args.tmp_folder,
            model_name=args.fid_model_name,
            batch_size=args.fid_batch_size)

    # Generating initial samples.
    if args.data_checkpoint_path != '':
        logging.info(
            f'Loading data checkpoint from {args.data_checkpoint_path}')
        samples, additional_info = load_samples(args.data_checkpoint_path)
        if args.data_checkpoint_step < 0:
            raise ValueError('data_checkpoint_step should be >= 0')
        start_t = args.data_checkpoint_step + 1
    else:
        logging.info('Generating initial samples')
        samples, additional_info = api.image_random_sampling(
            prompts=args.initial_prompt,
            num_samples=args.num_samples_schedule[0],
            size=args.image_size)
        log_samples(
            samples=samples,
            additional_info=additional_info,
            folder=f'{args.result_folder}/{0}',
            plot_images=args.plot_images)
        if args.data_checkpoint_step >= 0:
            logging.info('Ignoring data_checkpoint_step')
        start_t = 1

    if args.compute_fid:
        logging.info('Computing FID')
        fid = compute_fid(
            samples=samples,
            tmp_folder=args.tmp_folder,
            num_fid_samples=args.num_fid_samples,
            dataset_res=args.private_image_size,
            dataset=args.fid_dataset_name,
            dataset_split=args.fid_dataset_split,
            model_name=args.fid_model_name,
            batch_size=args.fid_batch_size)
        logging.info(f'fid={fid}')
        log_fid(args.result_folder, fid, 0)

    for t in range(start_t, len(args.num_samples_schedule)):
        logging.info(f't={t}')
        assert samples.shape[0] % private_num_classes == 0
        num_samples_per_class = samples.shape[0] // private_num_classes

        if args.lookahead_degree == 0:
            packed_samples = np.expand_dims(samples, axis=1)
        else:
            logging.info('Running image variation')
            packed_samples = api.image_variation(
                images=samples,
                additional_info=additional_info,
                num_variations_per_image=args.lookahead_degree,
                size=args.image_size,
                variation_degree=args.variation_degree_schedule[t])

        packed_features = []
        logging.info('Running feature extraction')
        for i in range(packed_samples.shape[1]):
            sub_packed_features = extract_features(
                data=packed_samples[:, i],
                tmp_folder=args.tmp_folder,
                model_name=args.feature_extractor,
                res=args.private_image_size,
                batch_size=args.feature_extractor_batch_size)
            logging.info(
                f'sub_packed_features.shape: {sub_packed_features.shape}')
            packed_features.append(sub_packed_features)
        packed_features = np.mean(packed_features, axis=0)

        logging.info('Computing histogram')
        count = []
        for class_i, class_ in enumerate(private_classes):
            sub_count, sub_clean_count = dp_nn_histogram(
                public_features=packed_features[
                    num_samples_per_class * class_i:
                    num_samples_per_class * (class_i + 1)],
                private_features=all_private_features[
                    all_private_labels == class_],
                noise_multiplier=args.noise_multiplier,
                num_nearest_neighbor=args.num_nearest_neighbor,
                mode=args.nn_mode,
                threshold=args.count_threshold)
            log_count(
                sub_count,
                sub_clean_count,
                f'{args.result_folder}/{t}/count_class{class_}.npz')
            count.append(sub_count)
        count = np.concatenate(count)
        for class_i, class_ in enumerate(private_classes):
            visualize(
                samples=samples[
                    num_samples_per_class * class_i:
                    num_samples_per_class * (class_i + 1)],
                packed_samples=packed_samples[
                    num_samples_per_class * class_i:
                    num_samples_per_class * (class_i + 1)],
                count=count[
                    num_samples_per_class * class_i:
                    num_samples_per_class * (class_i + 1)],
                folder=f'{args.result_folder}/{t}',
                suffix=f'class{class_}')

        logging.info('Generating new indices')
        assert args.num_samples_schedule[t] % private_num_classes == 0
        new_num_samples_per_class = (
            args.num_samples_schedule[t] // private_num_classes)
        new_indices = []
        for class_i in private_classes:
            sub_count = count[
                num_samples_per_class * class_i:
                num_samples_per_class * (class_i + 1)]
            sub_new_indices = np.random.choice(
                np.arange(num_samples_per_class * class_i,
                          num_samples_per_class * (class_i + 1)),
                size=new_num_samples_per_class,
                p=sub_count / np.sum(sub_count))
            new_indices.append(sub_new_indices)
        new_indices = np.concatenate(new_indices)
        new_samples = samples[new_indices]
        new_additional_info = additional_info[new_indices]

        logging.info('Generating new samples')
        new_new_samples = api.image_variation(
            images=new_samples,
            additional_info=new_additional_info,
            num_variations_per_image=1,
            size=args.image_size,
            variation_degree=args.variation_degree_schedule[t])
        new_new_samples = np.squeeze(new_new_samples, axis=1)
        new_new_additional_info = new_additional_info

        if args.compute_fid:
            logging.info('Computing FID')
            new_new_fid = compute_fid(
                new_new_samples,
                tmp_folder=args.tmp_folder,
                num_fid_samples=args.num_fid_samples,
                dataset_res=args.private_image_size,
                dataset=args.fid_dataset_name,
                dataset_split=args.fid_dataset_split,
                model_name=args.fid_model_name,
                batch_size=args.fid_batch_size)
            logging.info(f'fid={new_new_fid}')
            log_fid(args.result_folder, new_new_fid, t)

        samples = new_new_samples
        additional_info = new_new_additional_info
        log_samples(
            samples=samples,
            additional_info=additional_info,
            folder=f'{args.result_folder}/{t}',
            plot_images=args.plot_images)


if __name__ == '__main__':
    main()
