export CUDA_VISIBLE_DEVICES=2
exp_name='outputs/mipnerf360/r4_12v'
scenes=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")
dataset_path='../nerfs/data/mipnerf360'
n_views=12

for scene in "${scenes[@]}"
do
  echo "Training on $scene..."
  python train.py -s $dataset_path/$scene/ \
    -m $exp_name/$scene/ \
    --eval -r 4 \
    --n_views $n_views \
    --ip 127.0.0.03 \
    --pcd_path $dataset_path/$scene/multiview_pcd/${n_views}_views/twings_init_pcd.ply \
    --depth_loss --depth_weight 0.03 --depth_pseudo_weight 0.4
    
  echo "Rendering $scene..."
  python render.py -m $exp_name/$scene -r 4 
done

# Compute metrics for all scenes
python metric.py --path $exp_name