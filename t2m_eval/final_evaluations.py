import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from motion_loaders.model_motion_loaders import get_motion_loader_for_dechoi_eval
from utils.metrics import *
from networks.evaluator_wrapper import EvaluatorModelWrapper
from collections import OrderedDict
from utils.utils import *

from options.train_options import TrainTexMotMatchOptions

from os.path import join as pjoin

from vis_skeleton_motion import show3Dpose_animation 
from data.omomo_dataset import CanoObjectTrajDataset 
from utils.word_vectorizer import WordVectorizer


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_ROOT = Path(os.environ.get("DECHOI_PROCESSED_DATA", REPO_ROOT / "data" / "processed_data"))
GLOVE_ROOT = Path(os.environ.get("DECHOI_GLOVE_ROOT", REPO_ROOT / "glove_840B"))
RESULTS_ROOT = Path(os.environ.get("DECHOI_EVAL_RESULTS_ROOT", REPO_ROOT / "res_npz_files"))
REBUTTAL_RESULTS_ROOT = Path(
    os.environ.get("DECHOI_EVAL_REBUTTAL_ROOT", REPO_ROOT / "res_npz_files_rebuttal")
)

def plot_t2m(data, save_dir, ds):
    # data: BS X 2 X T X D 
    num_steps = data.shape[2]
   
    # data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        # joint_data = data[i][:, :, :24*3].reshape(-1, num_steps, 24, 3) # 2 X T X 24 X 3
        # joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        # joint = ds.de_normalize_jpos_min_max(joint_data) # 2 X T X 24 X 3 
        joint_data = data[i][:, :, :24*3] # 2 X T X 72 
        joint = ds.de_normalize_jpos_mean_std(joint_data) # 2 X T X 72 
        joint = joint.reshape(-1, num_steps, 24, 3) # 2 X T X 24 X 3 
        save_path = pjoin(save_dir, '%02d.mp4'%(i))
        # plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)
        show3Dpose_animation(joint.detach().cpu().numpy(), ds.parents, save_path) 

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict

def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    replication_times = 1 
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file):
    with open(log_file, 'w') as f:
        # all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
        #                            'R_precision': OrderedDict({}),
        #                            'FID': OrderedDict({}),
        #                            'Diversity': OrderedDict({}),
        #                            'MultiModality': OrderedDict({})})
        
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   })
        replication_times = 1 
        for replication in range(replication_times):
            motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(np.array(values))
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)

def check_vis(save_dir):
    w_vectorizer = WordVectorizer(str(GLOVE_ROOT), 'our_vab')

    val_dataset = CanoObjectTrajDataset(train=False, data_root_folder=str(PROCESSED_DATA_ROOT), \
                word_vectorizer=w_vectorizer) 
    
    motion_loaders = {}
    for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
        motion_loader = motion_loader_getter()
        motion_loaders[motion_loader_name] = motion_loader
       
    motion_loaders['ground_truth'] = gt_loader
    for motion_loader_name, motion_loader in motion_loaders.items():
        for idx, batch in enumerate(motion_loader):
            if not (idx % 4 == 0):
                continue 

            word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens, tokens = batch
            motions = motions[:, :m_lens[0]] # BS X T X 72 
            print('-----%d-----'%idx)
            print(captions)
            print(tokens)
            print(sent_lens)
            print(m_lens)

            ani_save_path = pjoin(save_dir, 'animation', '%02d'%(idx))
            os.makedirs(ani_save_path, exist_ok=True)
           
            plot_t2m(motions[:8, None], pjoin(ani_save_path, '%s' % (motion_loader_name)),
                          val_dataset)


if __name__ == '__main__':
    eval_motion_loaders = {
        'DecHOI w classifier guidance': lambda: get_motion_loader_for_dechoi_eval(
            REBUTTAL_RESULTS_ROOT / 'dechoi_perturb_mean',
            batch_size,
        ), 

        'DecHOI w recon x0': lambda: get_motion_loader_for_dechoi_eval(
            REBUTTAL_RESULTS_ROOT / 'dechoi_recon_guide_w_x0',
            batch_size,
        ), 

        'DecHOI': lambda: get_motion_loader_for_dechoi_eval(
            RESULTS_ROOT / 'dechoi',
            batch_size,
        ), 
    }

    batch_size = 32
    diversity_times = 300 

    gt_loader = get_motion_loader_for_dechoi_eval(RESULTS_ROOT / 'gt', batch_size)
    parser = TrainTexMotMatchOptions()
    wrapper_opt = parser.parse()

    device_id = 0
    wrapper_opt.device = torch.device('cuda:%d'%device_id if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    log_file = os.environ.get("DECHOI_EVAL_LOG_FILE", "./t2m_evaluation_dechoi.log")
    evaluation(log_file)
   
