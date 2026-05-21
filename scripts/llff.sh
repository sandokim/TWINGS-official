export CUDA_VISIBLE_DEVICES=1
exp_name='outputs/nerf_llff_data/r8_3v'
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
dataset_path='../nerfs/data/nerf_llff_data'
n_views=3

for scene in "${scenes[@]}"
do
  echo "Training on $scene..."
  python train.py -s $dataset_path/$scene/ \
    -m $exp_name/$scene \
    --eval -r 8 \
    --n_views $n_views \
    --ip 127.0.0.02 \
    --pcd_path $dataset_path/$scene/multiview_pcd/${n_views}_views/twings_init_pcd.ply \
    --depth_loss --depth_weight 0.03 --depth_pseudo_weight 0.1
    
  echo "Rendering $scene..."
  python render.py -m $exp_name/$scene -r 8
done

# Compute metrics for all scenes
python metric.py --path $exp_name