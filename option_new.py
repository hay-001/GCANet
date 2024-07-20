import argparse

parser = argparse.ArgumentParser()

# data parameters
parser.add_argument('--num_primitives',
                    type=int,
                    default=7,
                    help='num_primitives 7 or 10')
parser.add_argument('--mode',
                    type=int,
                    default=5,
                    help='0 only input pc; 3 only input pc, but predict normal; 5 input pc and normal')
parser.add_argument('--ablation',
                    type=bool,
                    default=False,
                    help='ablation for offset module')
parser.add_argument('--using_set_aggr',
                    type=bool,
                    default=False,
                    help='using_set_aggr')
parser.add_argument('--model_dict',
                    type=str,
                    # default='models.dgcnn-hais-concat-aff',
                    # default='models.dgcnn-hais-concat',#our
                    # default='models.dgcnn-hais-concat-direct-4', #-4our
                    # default='models.dgcnn-hais-concat-cos',
                    default='models.dgcnn-hais-concat-right-ab-inst',#our
                    help='model file name')
parser.add_argument('--checkpoint_path',
                    # default=None,
                    # default='log/dgcnn-hais-6-concat-tryagain/checkpoint_max_miou, miou:0.785,type_miou:0.904_eval27.tar',#our
                    # default='log/dgcnn-hais-10-concat-offset20/checkpoint_max_type_miou:0.798,type_miou:0.899_eval29.tar',
                    # default='log/dgcnn-hais-10-concat-cos/checkpoint_max_type_miou:0.804,type_miou:0.901_eval47.tar',
                    # default='log/direction4-3/checkpoint_max_miou, miou:0.811,type_miou:0.908_eval54.tar',
                    # default='log/direction4-2/checkpoint_max_miou, miou:0.809,type_miou:0.911_eval55.tar',#our
                    # default='log/direction4-3/checkpoint_max_miou, miou:0.811,type_miou:0.908_eval54.tar',
                    default='log/dgcnn-hais-6-concat-right-ab-inst/checkpoint_max_miou, miou:0.75,type_miou:0.778_eval1.tar',
                    help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir',
                    # default='log/dgcnn-hais-6-concat-without-normal',
                    # default='log/dgcnn-hais-6-concat-ablation-withoutoffset',
                    # default='log/dgcnn-hais-6-concat-ablation-withoutHCA',
                    # default='log/dgcnn-hais-6-concat-tryagain',
                    # default='log/dgcnn-hais-6-concat-offset20',
                    # default='log/dgcnn-hais-10-concat-offset20',
                    # default='log/dgcnn-hais-10-concat-cos',
                    # default='log/dgcnn-hais-6-concat-softgroup',
                    default='log/dgcnn-hais-6-concat-right-ab-inst',
                    help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--resultsSave',
                    type=bool,
                    default=False,
                    help='resultsSavePath')


parser.add_argument('--val_skip',
                    type=int,
                    default=5,
                    help='only test sub dataset')
parser.add_argument('--train_skip',
                    type=int,
                    default=1,
                    help='only train sub dataset')


parser.add_argument('--data_path', type=str, default='/opt/data/common/ABC/')
parser.add_argument('--dataset', type=str, default='ABC')
parser.add_argument('--train_dataset',
                    type=str,
                    default='train_data.txt',
                    help='file name for the list of object names for training')
parser.add_argument('--test_dataset',
                    type=str,
                    default='test_data.txt',
                    # default='real_data.txt',
                    help='file name for the list of object names for testing')
parser.add_argument('--batch_size', type=int, default=3)#default3,8
parser.add_argument('--vis',
                    action='store_true',
                    help='whether do the visualization')
parser.add_argument('--vis_dir',
                    type=str,
                    default=None,
                    help='visualization directory')
parser.add_argument('--eval', 
                    action='store_true', 
                    help='evaluate iou error')
parser.add_argument('--debug',
                    action='store_true',
                    help='whether switch to debug module')
parser.add_argument('--MEAN_SHIFT_STEP',
                    type=int,
                    default=5,
                    help='whether switch to debug module')


# training parameters
parser.add_argument('--max_epoch',
                    type=int,
                    default=200,
                    help='Epoch to run [default: 180]')
parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-3,
                    help='Initial learning rate [default: 0.001]')
parser.add_argument('--optimizer',
                    type=str,
                    default='adam',
                    help='[adam, sgd]')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0,
                    help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step',
                    type=int,
                    default=20,
                    help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate',
                    type=float,
                    default=0.5,
                    help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps',
                    default='20,30,40',#25,40,50
                    help='When to decay the learning rate (in epochs) [default: 30]')
parser.add_argument('--lr_decay_rates',
                    default='0.1,0.1,0.1',
                    help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--lr_decay_rate',
                    type=float,
                    default=0.1,
                    help='Decay rates for lr decay')
parser.add_argument('--loss_class',
                    type=str,
                    default='frpn',
                    help='loss functions; f:embedding loss; r:primitive loss;\
                          p:parameter loss, n:normal loss')

parser.add_argument('--train_fold',
                    type=int,
                    default=1)
parser.add_argument('--eval_interval',
                    type=int,
                    default=1,
                    help='evaluation interval')
parser.add_argument('--save_interval',
                    type=int,
                    default=2,
                    help='save specific checkpoint interval')
parser.add_argument('--augment',
                    type=int,
                    default=0,
                    help='whether do data augment')
parser.add_argument('--if_normal_noise',
                    type=int,
                    default=0,
                    help='whether do normal noise')
parser.add_argument('--optimize',
                    type=int,
                    default=0,
                    help='0: optimize feat loss; 1:optimize miou')







# model parameters
parser.add_argument('--gpu',
                    type=str,
                    default='0',
                    help='gpu number')
parser.add_argument('--not_load_model',
                    action='store_true',
                    help='whether load model from checkpoint')
parser.add_argument('--sigma',
                    type=float,
                    default=0.8,
                    help='affinity matrix hyper paramter')
parser.add_argument('--normal_sigma',
                    type=float,
                    default=0.1,
                    help='normal difference affinity matrix hyper paramter')
parser.add_argument('--out_dim',
                    type=int,
                    default=64,
                    help='output feature dimension')
parser.add_argument('--type_weight',
                    type=float,
                    default=2,#1.2
                    help='type loss weight')
parser.add_argument('--param_weight',
                    type=float,
                    default=0.1,#0.1
                    help='parameter loss weight')
parser.add_argument('--normal_weight',
                    type=float,
                    default=1.0,
                    help='normal loss weight')
parser.add_argument('--edge_knn',
                    type=int,
                    default=50,
                    help='k nearest neighbor of normal')

parser.add_argument('--feat_ent_weight',
                    type=float,
                    default=1.70,#1.7
                    help='network feature entropy weight')
parser.add_argument('--dis_ent_weight',
                    type=float,
                    default=1.10,
                    help='primitive distance entropy weight')
parser.add_argument('--edge_ent_weight',
                    type=float,
                    default=1.23,
                    help='edge boundary entropy weight')
parser.add_argument('--topK',
                    type=int,
                    default=10,
                    help='the number of eigenvectors used')
parser.add_argument('--edge_topK',
                    type=int,
                    default=12,
                    help='the number of eigenvectors edge feature used')
parser.add_argument('--bandwidth',
                    type=float,
                    default=0.85,
                    help='kernl bandwidth')
parser.add_argument('--backbone', 
                    type=str, 
                    default='DGCNN')

def build_option():
    FLAGS = parser.parse_args()
    return FLAGS
