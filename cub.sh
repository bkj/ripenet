export CUDA_VISIBLE_DEVICES=1

iter="cub_4"
python cub.py \
    --algorithm hyperband \
    --outpath _results/cub/$iter \
    --child-lr-init 0.01 \
    --epochs 1000 \
    --num-ops 8 \
    --num-nodes 3 \
    --seed 678 \
    --population-size 32 \
    --child-lr-schedule sgdr \
    --child-sgdr-period-length 20 \
    --child-sgdr-t-mult 1 \
    --hyperband-halving \
    --hyperband-resample \
    --controller-eval-interval 1 \
    --controller-train-interval 20 \
    --child-train-paths-per-epoch 47 # One epoch w/ 128 batch size


iter="reinforce_2"
python cub.py \
    --algorithm reinforce \
    --outpath _results/cub/$iter \
    --child-lr-init 0.001 \
    --epochs 1000 \
    --num-ops 6 \
    --num-nodes 2 \
    --seed 678 \
    --entropy-penalty 0.05 \
    --child-lr-schedule sgdr \
    --child-sgdr-period-length 20 \
    --child-sgdr-t-mult 1 \
    --controller-eval-interval 1 \
    --controller-train-interval 1

# --
# Baselines


# lr=0.2 constant
# 
# AdaptiveMultiPool2d(output_size=(1, 1)),
# Flatten(),
# nn.BatchNorm1d(num_features),
# nn.Linear(in_features=num_features, out_features=num_classes),
# 
# {'stage': 'classifier_precomputed', 'epoch': 99, 'train_debias_loss': 0.00032275900316047486, 
#   'valid_loss': 2.2461216764135674, 'valid_acc': 0.62858129099068, 
#   'val_test_loss': 2.2287328528321306, 'val_test_acc': 0.6254746289264757, 
#   'test_test_loss': 2.225614817246147, 'test_test_acc': 0.6316879530548843}


# Same, but w/ PreActBlock before -- need to do these benchmarks more thoroughly
# 
# lr=0.2 constant
# {'stage': 'classifier_precomputed', 'epoch': 49, 'train_debias_loss': 0.0003051913195044102, 
#   'valid_loss': 1.9309451442498426, 'valid_acc': 0.6439420089748015, 
#   'val_test_loss': 1.9219780745713606, 'val_test_acc': 0.6434242319641008, 
#   'test_test_loss': 1.9307441452275151, 'test_test_acc': 0.6444597859855022}
# 
# lr=0.001 sgdr (period=10, tmult=1)
# {'stage': 'classifier_precomputed', 'epoch': 49, 'train_debias_loss': 0.9247510357391342, 'valid_loss': 1.6575793840072968, 'valid_acc': 0.5973420780117363, 'val_test_loss': 1.642517986504928, 'val_test_acc': 0.5954435623058336, 'test_test_loss': 1.6683286609856978, 'test_test_acc': 0.5992405937176389}
#
# lr=0.01 sgdr (period=10, tmult=1)
# {'stage': 'classifier_precomputed', 'epoch': 49, 'train_debias_loss': 0.014071010370408683, 'valid_loss': 1.3808545516087458, 'valid_acc': 0.6430790472903003, 'val_test_loss': 1.3523170157619144, 'val_test_acc': 0.6486020020711081, 'test_test_loss': 1.4257928677227185, 'test_test_acc': 0.6375560925094926}
# 
# lr=0.1 sgdr (period=10, tmult=1)
# {'stage': 'classifier_precomputed', 'epoch': 49, 'train_debias_loss': 0.001318921230616561, 
#   'valid_loss': 1.5961712913198784, 'valid_acc': 0.644977562996203, 
#   'val_test_loss': 1.5686492660771245, 'val_test_acc': 0.6465308940283051, 
#   'test_test_loss': 1.610365092754364, 'test_test_acc': 0.6434242319641008}
