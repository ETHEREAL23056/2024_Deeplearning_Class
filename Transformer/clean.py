import argparse

from utils import *


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--cls_data', action='store_true', help='clean classify data')
    parser.add_argument('--gen_data', action='store_true', help='clean generate data')
    parser.add_argument('--show_charts', action='store_true', help='show static charts')
    parser.add_argument('--sample_rate', type=float, default=.01, help='sample rate')

    args = parser.parse_args()

    # 导入数据集
    data = pd.read_json('./dataset/News_Category_Dataset_v3.json', lines=True)

    # 同类合并
    data = merge_same_category(data=data)
    print(set(data['category']))

    # 去重
    data = rm_duplicated(data=data)

    # 编码统计
    data = label_encoder(data=data)
    if args.show_charts:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.histplot(data['sd_len'], kde=True, ax=axes[0], bins=30)
        axes[0].set_title('Description Number of Words')

        sns.histplot(data['hl_len'], kde=True, ax=axes[1], bins=30)
        axes[1].set_title('Headline Number of Words')

        plt.tight_layout()
        plt.title("encode statistic")
        plt.show()

    # 去除表情符号
    data['emojis'], data['short_description'] = zip(*data['short_description'].apply(extract_and_remove_emojis))

    emoji_count = data['emojis'].apply(lambda x: len(x)).sum()

    print(f"行中包含表情符号的总数量: {emoji_count}")

    emoji_rows = data[data['emojis'].apply(lambda x: len(x) > 0)]
    print("包含表情符号的行中的表情符号及剔除后的文本：")
    for index, row in emoji_rows.iterrows():
        print(f"行 {index} 的表情符号: {row['emojis']}, 剔除后的文本: {row['short_description']}")

    # 将标题给添加短描述中——仅对于分类任务
    if args.cls_data:
        data['short_description'] = data['headline'] + data['short_description']

    if args.show_charts:
        sns.distplot(data['hl_len'] + data['sd_len'])
        plt.title('Short Description Number of Words')
        plt.show()

    #  类别分布
    category_counts = data['category'].value_counts()
    print(category_counts)
    if args.show_charts:
        plt.figure(figsize=(12, 8))
        sns.barplot(x=category_counts.index, y=category_counts.values)
        plt.xticks(rotation=90)
        plt.title('News Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('Number of Articles')
        plt.show()

    # 时间趋势
    data['date'] = pd.to_datetime(data['date'])
    monthly_counts = data.groupby(data['date'].dt.to_period('M')).size()
    print(monthly_counts)
    if args.show_charts:
        monthly_counts.plot(kind='line', figsize=(12, 6))
        plt.title('News Articles Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.show()

    # 关键词统计
    keywords_df = group_keyword(data=data)
    categories = keywords_df['Category'].unique()
    if args.show_charts:
        n_categories = len(categories)
        for i in range(0, n_categories, 3):
            fig, axes = plt.subplots(1, 3, figsize=(12, 5))

            for j, category in enumerate(categories[i:i + 3]):
                group = keywords_df[keywords_df['Category'] == category]

                group = group.sort_values(by='Frequency', ascending=False).head(10)

                ax = axes[j]
                ax.barh(group["Keyword"], group["Frequency"], color='skyblue')
                ax.set_xlabel("Frequency")
                ax.set_ylabel("Keyword")
                ax.set_title(f"Top 10 Keywords for Category: {category}")
                ax.invert_yaxis()
            for j in range(len(categories[i:i + 3]), 3):
                axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    # 压缩数据集
    data = data.sample(frac=args.sample_rate, random_state=42).reset_index(drop=True)

    # 比例采样
    data_resampled = over_sample(data=data)
    train_data, val_data, test_data = data_split(data=data_resampled, base_label="categoryEncoded")

    print("Training set distribution:")
    print(train_data["categoryEncoded"].value_counts())

    print("\nValidation set distribution:")
    print(val_data["categoryEncoded"].value_counts())

    print("\nTest set distribution:")
    print(test_data["categoryEncoded"].value_counts())

    # 保存数据
    path = "./dataset/{name}.csv"
    if args.cls_data:
        columns_to_save = ["categoryEncoded", "short_description"]
        save_data(data=train_data, path=path.format(name="train_data"), select_col=columns_to_save)
        save_data(data=test_data, path=path.format(name="test_data"), select_col=columns_to_save)
        save_data(data=val_data, path=path.format(name="val_data"), select_col=columns_to_save)
    else:
        columns_to_save = ["headline", "short_description"]
        save_data(data=train_data, path=path.format(name="train_data_gen"), select_col=columns_to_save)
        save_data(data=test_data, path=path.format(name="test_data_gen"), select_col=columns_to_save)
        save_data(data=val_data, path=path.format(name="val_data_gen"), select_col=columns_to_save)


main()
