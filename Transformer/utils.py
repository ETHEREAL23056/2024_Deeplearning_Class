import re
import warnings

import bert_score
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
import seaborn as sns
from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, classification_report, \
    confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nltk.metrics.distance import edit_distance
from rouge import Rouge

from model import TextClsDataset, TransformerForClassification, device, TransformerForGeneration, TextGenDataset


# import nltk
# nltk.download('wordnet')


def merge_same_category(data: pd.DataFrame) -> pd.DataFrame:
    data.category = data.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
    data.category = data.category.map(lambda x: "ARTS & CULTURE" if x == "CULTURE & ARTS" else x)
    print(f"The dataset contains {data.category.nunique()} unique categories")
    return data


def rm_duplicated(data) -> pd.DataFrame:
    duplicates = data[data.duplicated(keep=False)]
    if not duplicates.empty:
        unique_duplicates_count = duplicates.drop_duplicates().shape[0]
        print(f"存在 {unique_duplicates_count} 种重复的行，重复的内容如下：")
        print(duplicates)
    else:
        print("无重复行")
    # 剔除重复的行，仅保留唯一行
    data = data.drop_duplicates()
    print(f"\n去除重复行后，剩余 {len(data)} 行数据。")

    duplicates = data[data.duplicated(keep=False)]  # keep=False 保留所有重复项
    if not duplicates.empty:
        unique_duplicates_count = duplicates.drop_duplicates().shape[0]
        print(f"存在 {unique_duplicates_count} 种重复的行，重复的内容如下：")
        print(duplicates)
    else:
        print("已无重复行")
    return data


def label_encoder(data: pd.DataFrame) -> pd.DataFrame:
    encoder = LabelEncoder()
    data['categoryEncoded'] = encoder.fit_transform(data['category'])
    label_mapping = {index: label for index, label in enumerate(encoder.classes_)}
    print("Category Encoding Mapping:")
    for encoded_value, category in label_mapping.items():
        print(f"{encoded_value} -> {category}")
    data['sd_len'] = data['short_description'].apply(lambda x: len(str(x).split()))
    data['hl_len'] = data['headline'].apply(lambda x: len(str(x).split()))
    data[['categoryEncoded', 'sd_len', 'hl_len']].describe()
    return data


def extract_and_remove_emojis(text: str) -> (str, str):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号（emoticons）
        "\U0001F300-\U0001F5FF"  # 符号和象形文字（symbols & pictographs）
        "\U0001F680-\U0001F6FF"  # 交通和地图符号（transport & map symbols）
        "\U0001F700-\U0001F77F"  # 炼金符号（alchemical symbols）
        "\U0001F780-\U0001F7FF"  # 几何形状扩展（Geometric Shapes Extended）
        "\U0001F800-\U0001F8FF"  # 补充箭头-C（Supplemental Arrows-C）
        "\U0001F900-\U0001F9FF"  # 补充符号和象形文字（Supplemental Symbols and Pictographs）
        "\U0001FA00-\U0001FA6F"  # 国际象棋符号（Chess Symbols）
        "\U0001FA70-\U0001FAFF"  # 符号和象形文字扩展-A（Symbols and Pictographs Extended-A）
        "\U00002702-\U000027B0"  # 字体符号（Dingbats）
        "\U000024C2-\U0001F251"  # 包围字符（Enclosed characters）
        "]+",  # 匹配一个或多个上述Unicode范围内的表情符号
        flags=re.UNICODE  # 启用Unicode匹配
    )

    # 提取表情符号
    emojis = emoji_pattern.findall(text)
    # 从文本中剔除表情符号
    cleaned_text = emoji_pattern.sub('', text)

    return emojis, cleaned_text


def group_keyword(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby('category')['short_description'].apply(lambda x: ' '.join(x)).reset_index()

    vectorizer = CountVectorizer(stop_words='english', max_features=10)

    # 提取每个类别的关键词
    category_keywords = {}
    for _, row in grouped.iterrows():
        category = row['category']
        text = row['short_description']
        X = vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
        frequencies = X.toarray().sum(axis=0)
        category_keywords[category] = list(zip(keywords, frequencies))

    keywords_df = pd.DataFrame([
        {"Category": category, "Keyword": keyword, "Frequency": freq}
        for category, keywords in category_keywords.items()
        for keyword, freq in keywords
    ])
    return keywords_df


def over_sample(data: pd.DataFrame) -> pd.DataFrame:
    ros = RandomOverSampler(random_state=42)

    X = data[['headline', 'short_description']]
    y = data['categoryEncoded']
    X_resampled, y_resampled = ros.fit_resample(X, y)
    data_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    return data_resampled


def data_split(data: pd.DataFrame, base_label: str, train_rate: float = .6, test_rate: float = .3, val_rate: float = .1,
               seed: int = 0) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_data, temp_data = train_test_split(
        data,
        test_size=(1 - train_rate),
        stratify=data[base_label],
        random_state=seed
    )
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(test_rate / (test_rate + val_rate)),
        stratify=temp_data[base_label],
        random_state=seed
    )
    return train_data, val_data, test_data


def save_data(data: pd.DataFrame, path: str, select_col: [str]) -> None:
    data[select_col].to_csv(path, index=False, encoding='utf-8')


def rm_null(dataset: pd.DataFrame, label: str) -> pd.DataFrame:
    dataset = dataset[dataset[label].notna()].reset_index(drop=True)
    return dataset


def predict_cls(model_path: str, test_path: str, save_path: str) -> (list, list, np.ndarray):
    BATCH_SIZE = 16
    MAX_LENGTH = 512
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    VOCAB_SIZE = tokenizer.vocab_size
    D_MODEL = 512
    D_FF = 1024
    MAX_SEQ_LENGTH = 512
    DROPOUT = .1
    NUM_CLASSES = 40

    # 导入测试数据
    test_dataset = pd.read_csv(test_path)

    # 去除空值
    test_dataset = test_dataset[test_dataset['short_description'].notna()].reset_index(drop=True)
    print(test_dataset.shape)

    # 评估模型
    test_data = TextClsDataset(test_dataset['short_description'], test_dataset['categoryEncoded'], tokenizer,
                               MAX_LENGTH)
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    model_info = model_path.split('_')
    NUM_HEADS = int(model_info[-2])
    NUM_LAYERS = int(model_info[-1])
    model = TransformerForClassification(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, MAX_SEQ_LENGTH, NUM_CLASSES,
                                         DROPOUT, device)
    model.load_state_dict(torch.load(f'./model/{model_path}.pth', map_location=torch.device(device)))

    pred_list, true_list, prob_list = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs, _ = model(input_ids)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            prob_list.append(probabilities)  # shape: [batch_size, num_classes]
            predicted_classes = torch.argmax(outputs, dim=1)
            pred_list += predicted_classes.cpu().tolist()
            true_list += labels.cpu().tolist()
    prob_list = np.concatenate(prob_list, axis=0)
    np.savez(save_path, true=true_list, pred=pred_list, probs=prob_list)
    return true_list, pred_list, prob_list


def show_weights_cls(model_path: str, test_input: str) -> None:
    MAX_LENGTH = 512
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(test_input, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)

    VOCAB_SIZE = tokenizer.vocab_size
    D_MODEL = 512
    D_FF = 1024
    MAX_SEQ_LENGTH = 512
    DROPOUT = .1
    NUM_CLASSES = 40

    model_info = model_path.split('_')
    NUM_HEADS = int(model_info[-2])
    NUM_LAYERS = int(model_info[-1])
    model = TransformerForClassification(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, MAX_SEQ_LENGTH, NUM_CLASSES,
                                         DROPOUT, device)
    model.load_state_dict(torch.load(f'./model/{model_path}.pth', map_location=torch.device(device)))

    model.eval()
    with torch.no_grad():
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].squeeze().to(device)
        _, attention_weights = model(input_ids)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        non_padding_tokens = [token for token, mask in zip(tokens, attention_mask) if mask != 0]
        print("Tokens corresponding to input_ids (excluding padding):")
        for idx, token in enumerate(non_padding_tokens):
            print(f'idx: {idx}, token: {token}')
        input_length = (attention_mask != 0).sum().item()
        for layer in range(NUM_LAYERS):
            print(f"--------------Layer {layer + 1}:--------------")
            for head in range(NUM_HEADS):
                attention = attention_weights[layer][0][head].cpu().numpy()  # (seq_len, seq_len)
                attention = attention[:input_length, :input_length]

                print(f"--------------Head {head + 1} Attention Weights:--------------")
                print(attention)

                # 绘制注意力热图
                plt.figure(figsize=(8, 8))
                plt.imshow(attention, cmap='viridis', aspect='auto')
                plt.colorbar(label="Attention Weight")
                plt.title(f"Attention Weights (Layer {layer + 1}, Head {head + 1})")
                plt.xlabel("Key Positions")
                plt.ylabel("Query Positions")
                plt.show()


def predict_gen(model_path: str, test_path: str, save_path: str) -> (list, list, np.ndarray):
    BATCH_SIZE = 16
    MAX_LENGTH = 512
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = tokenizer.vocab_size, tokenizer.vocab_size
    D_MODEL = 512
    D_FF = 1024
    MAX_SEQ_LENGTH = 512
    DROPOUT = .1

    # 导入测试数据
    test_dataset = pd.read_csv(test_path)

    # 去除空值
    test_dataset = test_dataset[test_dataset['short_description'].notna()].reset_index(drop=True)
    print(test_dataset.shape)

    # 评估模型
    test_data = TextGenDataset(test_dataset['short_description'], test_dataset['headline'], tokenizer, tokenizer,
                               MAX_LENGTH)
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model_info = model_path.split('_')
    NUM_HEADS = int(model_info[-2])
    NUM_LAYERS = int(model_info[-1])
    model = TransformerForGeneration(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF,
                                     MAX_SEQ_LENGTH, DROPOUT, device)
    model.load_state_dict(torch.load(f'./model/{model_path}.pth', map_location=torch.device(device)))

    generated_sentences, real_sentences = [], []
    model.eval()

    for batch_idx, (src, tgt) in enumerate(test_data_loader):
        src = src.to(device)
        tgt = tgt.to(device)

        for i in range(tgt.size(0)):
            generated_sentence = generate_sentence(model, src[i], tokenizer)
            generated_sentences.append(generated_sentence)
            real_sentence = tokenizer.decode(tgt[i].cpu().tolist(), skip_special_tokens=True)
            real_sentences.append(real_sentence)
            print('-------------------------------')
            print(f'real sentence: {real_sentence}')
            print(f'generate sentence: {generated_sentence}')

    np.savez(save_path, true=real_sentences, pred=generated_sentences)
    return real_sentences, generated_sentences


def generate_sentence(model, src, tokenizer, max_length=512):
    with torch.no_grad():
        start_token = tokenizer.cls_token_id
        sep_token = tokenizer.sep_token_id
        decoder_input = torch.tensor([[start_token]], device=device)
        src = src.unsqueeze(0)  # 将形状变为 (1, seq_len)

        for _ in range(max_length):
            output, _ = model(src, decoder_input)  # Shape: [1, seq_len, vocab_size]
            predicted_token = output.argmax(dim=-1)[:, -1]  # Shape: [1], select the last token
            decoder_input = torch.cat((decoder_input, predicted_token.unsqueeze(0)), dim=1)
            if predicted_token.item() == sep_token:
                break

        generated_sentence = tokenizer.decode(decoder_input.squeeze().tolist(), skip_special_tokens=True)
        return generated_sentence


def calculate_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    n_classes = len(classes)

    measure_result = classification_report(y_true, y_pred)
    print('measure_result = \n', measure_result)

    # 准确率（Accuracy）
    accuracy = accuracy_score(y_true, y_pred)

    # 本次试验集类别比重平衡,采用宏平均指标
    precision_none = precision_score(y_true, y_pred, average=None)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_none = recall_score(y_true, y_pred, average=None)
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_none = f1_score(y_true, y_pred, average=None)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # 四舍五入到两位小数
    precision_none_rounded = np.around(precision_none, decimals=2)
    recall_none_rounded = np.around(recall_none, decimals=2)
    f1_none_rounded = np.around(f1_none, decimals=2)

    print(f"Accuracy: {accuracy:.2f}")  # 准确率

    # 打印每个类别的精确率、召回率和F1分数
    for i in range(n_classes):
        print(f"Class {i}: Precision={precision_none_rounded[i]:.2f}, "
              f"Recall={recall_none_rounded[i]:.2f}, "
              f"F1 Score={f1_none_rounded[i]:.2f}")

    print(f"Macro-average Precision: {precision_macro:.2f}")  # 精确率
    print(f"Macro-average Recall: {recall_macro:.2f}")  # 召回率
    print(f"Macro-average F1 Score: {f1_macro:.2f}")  # F1 分数


def plot_confusion_matrix(y_true, y_pred, normalize=False, figsize=(15, 15)):
    # 获取类别名称（假设类别是连续整数）
    classes = np.unique(np.concatenate((y_true, y_pred)))

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'  # 归一化时使用浮点数格式
        print("Normalized confusion matrix")
    else:
        fmt = 'd'  # 不归一化时使用整数格式
        print('Confusion matrix, without normalization')

    # 使用Seaborn绘制热图
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # 调整布局以防止标签被裁剪
    plt.tight_layout()

    # 展示图表
    plt.show()


def plot_roc_curve(y_true, y_pred_prob):
    # 获取所有类别的唯一标签
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)

    # 将真实标签转换为二进制格式（独热编码）
    y_true_binarized = label_binarize(y_true, classes=unique_classes)

    # 初始化绘图
    plt.figure(figsize=(10, 7))

    # 存储每个类别的AUC值
    class_auc = {}

    # 绘制每个类别的ROC曲线
    for i, class_label in enumerate(unique_classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        class_auc[class_label] = roc_auc
        plt.plot(fpr, tpr)
    # plt.plot(fpr, tpr, label=f'Class {class_label}')  # 不在标签中显示AUC

    # 计算微平均ROC曲线和AUC
    false_positive_rate_micro, true_positive_rate_micro, _ = roc_curve(y_true_binarized.ravel(), y_pred_prob.ravel())
    roc_auc_micro = auc(false_positive_rate_micro, true_positive_rate_micro)
    class_auc['Micro-average'] = roc_auc_micro
    plt.plot(false_positive_rate_micro, true_positive_rate_micro, color='darkorange', linestyle='--',
             label='Micro-average')  # 不在标签中显示AUC

    # 添加对角线作为参考线（随机猜测的情况）
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    # 设置图表标题和轴标签
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Class Classification')

    # 显示图例和网格
    plt.legend(loc="lower right")
    plt.grid()

    # 展示图表
    plt.show()

    # 单独列出AUC
    print("\nAUC values:")
    for class_label, auc_value in class_auc.items():
        print(f"Class {class_label}: AUC = {auc_value:.2f}")


def cal_avg_bleu(true_labels: list[str], predicted_labels: list[str]) -> float:
    total_score = 0
    for true, pred in zip(true_labels, predicted_labels):
        true_sentence = [word_tokenize(true.lower())]  # 参考序列
        pred_sentence = word_tokenize(pred.lower())  # 生成序列
        # 使用平滑方法计算 BLEU
        smoothing_function = SmoothingFunction().method1
        total_score += sentence_bleu(true_sentence, pred_sentence, smoothing_function=smoothing_function)
    avg_score = total_score / len(true_labels)
    return avg_score


def cal_avg_rouge(true_labels: list[str], predicted_labels: list[str]) -> dict:
    rouge = Rouge()
    total_rouge, avg_rouge = {}, {}
    for true, pred in zip(true_labels, predicted_labels):
        scores = rouge.get_scores(true, pred, avg=True)
        for key in scores.keys():
            if key not in total_rouge.keys():
                total_rouge[key] = scores[key]
            else:
                for in_key in scores[key].keys():
                    total_rouge[key][in_key] += scores[key][in_key]
    for key in total_rouge.keys():
        avg_rouge[key] = total_rouge[key]
        for in_key in total_rouge[key].keys():
            avg_rouge[key][in_key] = total_rouge[key][in_key] / len(true_labels)
    return avg_rouge


def cal_avg_meteor(true_labels: list[str], predicted_labels: list[str]) -> float:
    total_score = 0
    for true, pred in zip(true_labels, predicted_labels):
        true_sentence = [word_tokenize(true.lower())]  # 参考序列
        pred_sentence = word_tokenize(pred.lower())  # 生成序列
        total_score += meteor_score(true_sentence, pred_sentence)
    avg_score = total_score / len(true_labels)
    return avg_score


def cal_avg_edit_distance(true_labels: list[str], predicted_labels: list[str]) -> float:
    total_distance = 0
    for true, pred in zip(true_labels, predicted_labels):
        distance = edit_distance(true, pred)
        total_distance += distance

    avg_distance = total_distance / len(true_labels)
    return avg_distance


def cal_avg_bert_score(true_labels: list[str], predicted_labels: list[str]) -> dict:
    total_score = {'precision': 0.0, 'recall': 0.0, 'F1 score': 0.0}
    warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")

    for true, pred in zip(true_labels, predicted_labels):
        P, R, F1 = bert_score.score([pred], [true], lang='en')

        total_score['precision'] += P.mean().item()
        total_score['recall'] += R.mean().item()
        total_score['F1 score'] += F1.mean().item()

    avg_score = {key: total_score[key] / len(true_labels) for key in total_score}

    return avg_score
