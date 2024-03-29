import argparse
import torch

#(args.backbone, args.setting, args.trainset, args.shared) #args.expname only makes sense if args.setting is not allgame

#if args.backbone e2e.. then the gradients flow through the backbone during training
def get_args():
    parser = argparse.ArgumentParser(description='Prtr')

    parser.add_argument(
        "--arch",
        choices=["standard", "resnet"],
        default="standard",
    )
    parser.add_argument(
        "--model",
        choices=["FPV_RECONBEV_CARLA", "FPV_BEV_CARLA", "BEV_VAE_CARLA", "BEV_LSTM_CARLA", "4STACK_VAE_ATARI", "3CHANRGB_VAE_ATARI101", "1CHAN_VAE_ATARI101", "3CHAN_VAE_ATARI", "1CHAN_VAE_ATARI", "1CHANLSTM_CONT_ATARI", "4STACK_CONT_ATARI", "DUAL_4STACK_CONT_ATARI", "3CHANLSTM_CONT_BEOGYM", "1CHAN_CONT_ATARI", "3CHAN_TCN_BEOGYM", "3CHAN_VIP_BEOGYM", "3CHAN_OVIP_BEOGYM", "1CHAN_VIP_ATARI", "4STACK_VIP_ATARI", "1CHAN_VEP_ATARI", "1CHAN_NVEP_ATARI", "1CHAN_TCN_ATARI", "1CHAN_SOM_ATARI", "1CHAN_OVIP_ATARI", "3CHAN_SOM_BEOGYM", "3CHAN_VEP_BEOGYM", "3CHAN_NVEP_BEOGYM", "3CHAN_VIP_BEOGYM", "FPV_BEV_TCN_CARLA", "FPV_TCN_CARLA"],
        default="e2e",
    )
    parser.add_argument(
        "--load_checkpoint", action="store_true", default=False, help="Load ckpt or not"
    )

    parser.add_argument(
        "--test", action="store_true", default=False, help="test mode or train mode"
    )
    
    parser.add_argument(
        "--save_dir", type=str, default="/lab/kiran/ckpts/pretrained/carla/", help="pretrained results"
    )
    parser.add_argument(
        "--tmodel_path", type=str, default="", help="pretrained results"
    ) 
    parser.add_argument(
        "--model_path", type=str, default="", help="pretrained results"
    )    
    parser.add_argument(
        "--texpname", type=str, default="all", help="pretrained results"
    )
    parser.add_argument(
        "--expname", type=str, default="all", help="pretrained results"
    )
    parser.add_argument(
        "--negtype", type=str, default="same", help="pretrained results"
    )
    parser.add_argument(
        "--maxseq", type=int, default=7500, help="pretrained results"
    )
    parser.add_argument(
        "--nepoch", type=int, default=1, help="pretrained results"
    )
    #201 originally
    parser.add_argument(
        "--nrounds", type=int, default=5, help="GPU ID"
    )
    parser.add_argument(
        "--min_len", type=float, default=20, help="GPU ID"
    )
    parser.add_argument(
        "--dthresh", type=float, default=-1, help="GPU ID"
    )
    parser.add_argument(
        "--max_len", type=float, default=4, help="GPU ID"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU ID"
    )
    parser.add_argument(
        "--sgamma", type=float, default=.01, help="GPU ID"
    )
    parser.add_argument(
        "--temperature", type=float, default=.1, help="GPU ID"
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=512, help="pretrained results"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=512, help="pretrained results"
    )
    parser.add_argument(
        "--kl_weight", type=float, default=0.01, help="pretrained results"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="pretrained results"
    )

    args = parser.parse_args()
    return args
