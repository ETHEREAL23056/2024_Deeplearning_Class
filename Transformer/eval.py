import argparse

from utils import *


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--model_name', type=str, default='', help='model name')
    parser.add_argument('--test_cls', action='store_true', help='get classify prediction')
    parser.add_argument('--test_gen', action='store_true', help='get generate prediction')
    parser.add_argument('--save_name', type=str, default='', help='prediction save name')
    parser.add_argument('--eval_name', type=str, default='', help='eval dataset name')
    parser.add_argument('--show_attention', action='store_true', help='show attention weights')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.test_cls or args.test_gen:
        if args.test_cls:
            predict_cls(args.model_name, './dataset/test_data.csv', f'./predictions/{args.save_name}.npz')
        else:
            predict_gen(args.model_name, './dataset/test_data_gen.csv', f'./predictions/{args.save_name}.npz')
    elif args.show_attention:
        show_weights_cls(args.model_name, "Brazil's Military Will Be Deployed To Guard Rio Tourist Sites")
    else:
        predictions = np.load(f'./predictions/{args.eval_name}.npz')
        trues = predictions['true']
        preds = predictions['pred']
        if args.eval_name.split('_')[-1] == 'gen':
            avg_bleu = cal_avg_bleu(true_labels=trues, predicted_labels=preds)
            print(f"----Average BLEU: {avg_bleu:.2f}----")
            avg_rouge = cal_avg_rouge(true_labels=trues, predicted_labels=preds)
            print("----Average Rouge:----")
            for key in avg_rouge.keys():
                print(f"{key}: {avg_rouge[key]}")
            avg_meteor = cal_avg_meteor(true_labels=trues, predicted_labels=preds)
            print(f"----Average METEOR: {avg_meteor:.2f}----")
            avg_distance = cal_avg_edit_distance(true_labels=trues, predicted_labels=preds)
            print(f"----Average Edit Distance: {avg_distance:.2f}----")
            # avg_bert_score = cal_avg_bert_score(true_labels=trues, predicted_labels=preds)
            # print("----Average BERT Score:----")
            # for key in avg_bert_score.keys():
            #     print(f"{key}: {avg_bert_score[key]}")
        else:
            probs = predictions['probs']
            calculate_metrics(y_true=trues, y_pred=preds)
            plot_confusion_matrix(y_true=trues, y_pred=preds)
            plot_roc_curve(y_true=trues, y_pred_prob=probs)


main()
