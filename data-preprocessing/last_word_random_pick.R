last_word_df <- read.csv(file.choose())

# 随机抽取 80 行
set.seed(123)  # 设置随机种子以确保结果可重复

sampled_rows <- last_word_df[sample(nrow(last_word_df), 80), ]

write.csv(sampled_rows, 'data/last_words_sampled_rows.csv')
