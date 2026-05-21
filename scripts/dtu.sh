export CUDA_VISIBLE_DEVICES=0
exp_name='outputs/DTU/r4_3v'
scenes=("scan8" "scan21" "scan30" "scan31" "scan34" "scan38" "scan40" "scan41" "scan45" "scan55" "scan63" "scan82" "scan103" "scan110" "scan114")
dataset_path='../nerfs/data/DTU/dtu_corgs'
n_views=3

for scene in "${scenes[@]}"
do
  echo "Training on $scene..."
  python train.py -s $dataset_path/$scene/ \
    -m $exp_name/$scene \
    --eval -r 4 \
    --n_views $n_views \
    --ip 127.0.0.01 \
    --pcd_path $dataset_path/$scene/multiview_pcd/${n_views}_views/twings_init_pcd.ply \
    --depth_loss --depth_weight 0.01 --depth_pseudo_weight 0.1

  bash ./scripts/copy_mask_dtu.sh "$exp_name"

  echo "Rendering $scene..."
  python render.py -m $exp_name/$scene -r 4

  python metrics_dtu.py \
  -m $exp_name/$scene
done

# Compute dtu masked metrics for all scenes
python metrics_means.py --exp_name $exp_name

