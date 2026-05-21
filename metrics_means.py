import os
import json
import numpy as np
import argparse

def get_suffixes(dataset):
    if dataset == 'nerf_llff_data':
        return ['r8_3v', 'r8_6v', 'r8_9v']
    elif dataset == 'mipnerf360':
        return ['r4_12v', 'r4_24v']
    elif dataset == 'DTU':
        return ['r4_3v', 'r4_6v', 'r4_9v']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_max_ours_key(data):
    ours_keys = [key for key in data.keys() if key.startswith('ours_')]
    if not ours_keys:
        return None
    return max(ours_keys, key=lambda x: int(x.split('_')[1]))

def search_and_collect_metrics(base_path, dataset):
    suffixes = get_suffixes(dataset)
    metrics = {suffix: {'PSNR': [], 'SSIM': [], 'LPIPS': [], 'AVGE': [], 'SSIM_sk': [], 'AVGE_sk': []} for suffix in suffixes}
    
    if dataset == 'DTU':
        # DTU
        for suffix in suffixes:
            suffix_path = os.path.join(base_path, suffix)
            if os.path.exists(suffix_path) and os.path.isdir(suffix_path):
                for scan_folder in os.listdir(suffix_path):
                    scan_path = os.path.join(suffix_path, scan_folder)
                    if os.path.isdir(scan_path):
                        result_file = os.path.join(scan_path, 'results_test_mask.json')
                        if os.path.exists(result_file):
                            with open(result_file, 'r') as f:
                                data = json.load(f)
                                max_ours_key = get_max_ours_key(data)
                                if max_ours_key:
                                    metrics[suffix]['PSNR'].append(data[max_ours_key]['PSNR'])
                                    metrics[suffix]['SSIM'].append(data[max_ours_key]['SSIM'])
                                    metrics[suffix]['LPIPS'].append(data[max_ours_key]['LPIPS'])
                                    metrics[suffix]['AVGE'].append(data[max_ours_key]['AVGE'])
                                    metrics[suffix]['SSIM_sk'].append(data[max_ours_key].get('SSIM_sk', None))
                                    metrics[suffix]['AVGE_sk'].append(data[max_ours_key].get('AVGE_sk', None))
    else:
        # mipnerf360 / nerf_llff_data
        excluded_folders = {'treehill', 'flowers'} if dataset == 'mipnerf360' else set()
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)
            if os.path.isdir(folder_path):
                for suffix in suffixes:
                    if suffix in folder_name:
                        for root, dirs, files in os.walk(folder_path):
                            # Exclude treehill and flowers folders
                            dirs[:] = [d for d in dirs if d not in excluded_folders]
                            if 'results.json' in files:
                                result_file = os.path.join(root, 'results.json')
                                with open(result_file, 'r') as f:
                                    data = json.load(f)
                                    max_ours_key = get_max_ours_key(data)
                                    if max_ours_key:
                                        metrics[suffix]['PSNR'].append(data[max_ours_key]['PSNR'])
                                        metrics[suffix]['SSIM'].append(data[max_ours_key]['SSIM'])
                                        metrics[suffix]['LPIPS'].append(data[max_ours_key]['LPIPS'])
    
    return metrics

def calculate_mean_metrics(metrics):
    mean_metrics = {}
    for suffix, metric_values in metrics.items():
        mean_metrics[suffix] = {
            metric: np.mean(values) for metric, values in metric_values.items() if values
        }
    return mean_metrics

def save_mean_metrics_as_json(mean_metrics, output_base_path):
    for suffix, metrics in mean_metrics.items():
        output_file = os.path.join(output_base_path, f"{suffix}_mean_metrics.json")
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect metrics and save mean results.")
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name (base directory containing dataset subfolders).')
    args = parser.parse_args()
    # DTU / mipnerf360 / nerf_llff_data
    dataset = 'DTU'
    base_path = os.path.join(*args.exp_name.split('/')[:-1])
    metrics = search_and_collect_metrics(base_path, dataset)
    mean_metrics = calculate_mean_metrics(metrics)
    save_mean_metrics_as_json(mean_metrics, base_path)

    print(f"Mean metrics have been saved in {base_path}.")
