import os
import torch
import numpy as np

from model import MODEL_DICT
from trainers import Trainer
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
from dataset import get_seq_dic, get_dataloder, get_rating_matrix

from item_graph_embedder import GCN, build_item_graph

def main():

    args = parse_args()  # 解析命令行参数

    # 检查并创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    log_path = os.path.join(args.output_dir, args.train_name + '.log')  # 在output文件夹内生成日志
    logger = set_logger(log_path)  # 设置日志

    set_seed(args.seed)  # 初始化随机种子，确保模型的训练结果可以复现

    check_path(args.output_dir)  # 检查并创建输出目录

    # 设置gpu设备，如果没有GPU就强制使用CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item, num_users = get_seq_dic(args)  # 读取用户交互序列（比如用户的点击、购买历史）
    args.item_size = max_item + 1     # 数据集的最大物品数, +1是为了索引
    args.num_users = num_users + 1    # 数据集的用户总数

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.data_name+'_same_target.npy')
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)  # 构造Pytorch的DataLoader，批量加载数据

    logger.info(str(args))

    # GCN --------------------------------------------------------
    # 构建图 & 图卷积
    edge_index = build_item_graph(seq_dic['user_seq'], args.item_size)

    gcn_model = GCN(num_nodes=args.item_size, hidden_dim=args.hidden_size).to('cuda' if args.cuda_condition else 'cpu')
    gcn_model.eval()
    with torch.no_grad():
        gcn_item_emb = gcn_model(edge_index.to(gcn_model.embedding.weight.device))  # shape: [num_items, hidden_dim]
    args.pretrained_item_emb = gcn_item_emb  # 把gcn embedding传入模型

    torch.save(gcn_item_emb.cpu(), os.path.join(args.output_dir, "item_gcn_emb.pt"))
    print("[INFO] GCN embedding saved.")

    # --------------------------------------------------------------

    gcn_emb_path = os.path.join(args.output_dir, "item_gcn_emb.pt")
    if os.path.exists(gcn_emb_path):
        args.pretrained_item_emb = torch.load(gcn_emb_path)
        print("[INFO] Loaded GCN embedding for initialization.")
    else:
        args.pretrained_item_emb = None
        print("[INFO] No GCN embedding found, using random init.")

    model = MODEL_DICT[args.model_type.lower()](args=args)  # 初始化模型，例如model = BSARec(args) 实例化BSARec模型
    logger.info(model)
    trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args, logger) # Trainer封装了训练train()、测试test()、验证valid()的逻辑

    # 在推荐系统中，valid_rating_matrix 和 test_rating_matrix 的作用是评估模型的推荐效果，主要用于 计算 HR@K、NDCG@K 等指标
    # 有了valid和test的稀疏交互矩阵，我们就可以快速计算指标
    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)  # 构造用户-物品评分矩阵，用于评估模型效果

    # 模型推理
    if args.do_eval:
        if args.load_model is None:
            logger.info(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)

            logger.info(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0)
            args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
            # torch.save(trainer.model.state_dict(), args.checkpoint_path)

    # 模型训练
    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):

            trainer.train(epoch)   # 从这一行开始训练
            scores, _ = trainer.valid(epoch)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)

    logger.info(args.train_name)
    logger.info(result_info)


if __name__ == '__main__':
    main()
