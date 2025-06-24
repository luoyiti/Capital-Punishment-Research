# 加载必要的包
library(readxl)
library(readr)
library(dplyr)
library(stringr)
library(caret)
library(randomForest)
library(ggplot2)
library(tidyr)
library(themis)   # 新增：用于SMOTE过采样
library(pROC)     # 新增：用于ROC分析
library(PRROC)    # 新增：用于PR曲线
library(tibble)
library(forcats)

# 1. 数据加载与预处理
# 读取主数据集
final_data <- read_excel("final_data_setv1.xlsx") %>%
  mutate(execution = as.character(execution)) %>%
  rename(
    prior_occupation = `prior occupation`,
    prior_prison_record01 = `prior_prison_record01`
  )

# 读取标注数据
annotated_data <- read_csv("last_words_sampled_rows.csv") %>%
  mutate(execution = as.character(execution)) %>%
  select(execution, penitence = label.penitence)

# 2. 增强特征工程
combined_data <- final_data %>%
  left_join(annotated_data, by = "execution") %>%
  mutate(
    education_years = str_extract(education_level, "\\d+") %>% 
      as.numeric() %>% 
      replace_na(0),
    
    race_simplified = case_when(
      str_detect(race, "White") ~ "White",
      str_detect(race, "Black") ~ "Black",
      str_detect(race, "Hispanic|Latino") ~ "Hispanic",
      TRUE ~ "Other"
    ) %>% factor(),
    
    occupation_count = ifelse(
      is.na(prior_occupation) | prior_occupation == "" | nchar(prior_occupation) == 0,
      0,
      str_count(prior_occupation, ",") + 1
    ),
    
    prior_record_factor = factor(prior_prison_record01, 
                                 levels = c(0, 1),
                                 labels = c("NoRecord", "HasRecord")),
    
    # 新增：创建交互特征
    edu_occupation = education_years * occupation_count,
    race_record = paste(race_simplified, prior_record_factor, sep = "_"),
    
    penitence_binary = factor(ifelse(penitence >= 0.5, "Penitent", "NotPenitent"),
                              levels = c("NotPenitent", "Penitent"))
  )

# 确保所有因子水平完整
combined_data <- combined_data %>%
  mutate(
    # 确保race_simplified有完整水平
    race_simplified = factor(race_simplified, 
                             levels = c("White", "Black", "Hispanic", "Other")),
    
    # 确保prior_record_factor有完整水平
    prior_record_factor = factor(prior_record_factor,
                                 levels = c("NoRecord", "HasRecord")),
    
    # 创建race_record因子并指定完整水平
    race_record = factor(
      paste(race_simplified, prior_record_factor, sep = "_"),
      levels = c(
        "White_NoRecord", "White_HasRecord",
        "Black_NoRecord", "Black_HasRecord",
        "Hispanic_NoRecord", "Hispanic_HasRecord",
        "Other_NoRecord", "Other_HasRecord"
      )
    )
  )

# 然后在数据集划分部分，确保训练数据和预测数据使用相同的因子水平
train_data <- combined_data %>% 
  filter(!is.na(penitence)) %>%
  select(education_years, race_simplified, prior_record_factor, 
         occupation_count, edu_occupation, race_record, penitence_binary) %>%
  drop_na() %>%
  # 确保所有因子水平保留
  mutate(across(where(is.factor), fct_drop))  # 仅保留实际存在的水平

# 修复预测数据集的创建
predict_data <- combined_data %>% 
  filter(is.na(penitence)) %>%
  select(execution, education_years, race_simplified, 
         prior_record_factor, occupation_count) %>%
  mutate(
    # 首先处理数值型特征
    education_years = replace_na(education_years, 0),
    occupation_count = replace_na(occupation_count, 0),
    
    # 然后处理因子特征 - 先转换为字符，再转换为因子
    race_simplified = as.character(race_simplified),
    race_simplified = ifelse(is.na(race_simplified), "Other", race_simplified),
    race_simplified = factor(
      race_simplified,
      levels = levels(train_data$race_simplified)
    ),
    
    prior_record_factor = as.character(prior_record_factor),
    prior_record_factor = ifelse(is.na(prior_record_factor), "NoRecord", prior_record_factor),
    prior_record_factor = factor(
      prior_record_factor,
      levels = levels(train_data$prior_record_factor)
    ),
    
    # 创建race_record特征
    race_record = paste(race_simplified, prior_record_factor, sep = "_"),
    race_record = factor(
      race_record,
      levels = levels(train_data$race_record)
    ),
    
    # 最后计算交互特征
    edu_occupation = education_years * occupation_count
  )

# 添加检查点
cat("预测数据集维度:", dim(predict_data), "\n")
if (nrow(predict_data) == 0) {
  stop("预测数据集为空！请检查数据预处理步骤。")
}

# 然后继续后续的预测和可视化代码...

# 3. 数据集划分
train_data <- combined_data %>% 
  filter(!is.na(penitence)) %>%
  select(education_years, race_simplified, prior_record_factor, 
         occupation_count, edu_occupation, race_record, penitence_binary) %>%  # 添加新特征
  drop_na()

# 检查类别分布
cat("类别分布:\n")
table(train_data$penitence_binary)
cat("\n 严重不平衡! 将应用SMOTE过采样\n")

# 4. 改进模型训练
set.seed(123)

# 创建类权重（惩罚多数类误判）
class_weights <- ifelse(train_data$penitence_binary == "Penitent", 1, 3)  # NotPenitent误判代价是3倍

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = prSummary,  # 改用PR曲线评估
  savePredictions = "final",
  sampling = "smote"  # 添加SMOTE过采样
)

# 训练带权重的随机森林
model <- train(
  penitence_binary ~ .,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  metric = "AUC",  # 使用PR AUC
  tuneLength = 5,   # 增加调参范围
  weights = class_weights,  # 添加代价敏感权重
  importance = TRUE  # 确保计算特征重要性
)

# 5. 阈值优化（基于F1分数）
# 提取交叉验证预测结果
cv_predictions <- model$pred %>%
  arrange(rowIndex) %>%
  select(Penitent, NotPenitent, obs, rowIndex)

# 寻找最佳阈值
thresholds <- seq(0.3, 0.7, by = 0.01)
f1_scores <- sapply(thresholds, function(thresh) {
  pred_class <- ifelse(cv_predictions$Penitent > thresh, "Penitent", "NotPenitent")
  pred_class <- factor(pred_class, levels = c("NotPenitent", "Penitent"))
  conf_mat <- confusionMatrix(pred_class, cv_predictions$obs, positive = "Penitent")
  conf_mat$byClass["F1"]
})

best_threshold <- thresholds[which.max(f1_scores)]
cat(sprintf("\n最佳分类阈值: %.2f (原始为0.50)", best_threshold))

# 修改特征重要性可视化部分（替换原来的代码）
# 6. 可视化分析
# 特征重要性
var_imp <- varImp(model)$importance

# 将重要性对象转换为数据框
var_imp_df <- as.data.frame(var_imp) %>%
  rownames_to_column("Feature") %>%
  # 重命名重要性列（根据实际列名调整）
  rename(Importance = contains("Penitent")[1])  # 选择第一个包含"Penitent"的列

feature_names <- c(
  "education_years" = "教育年限",
  "race_simplifiedHispanic" = "种族:拉丁裔",
  "race_simplifiedOther" = "种族:其他",
  "race_simplifiedWhite" = "种族:白人",
  "prior_record_factorHasRecord" = "犯罪记录",
  "occupation_count" = "职业数量",
  "edu_occupation" = "教育年限×职业数量",
  "race_recordBlack_HasRecord" = "黑人+有前科",
  "race_recordBlack_NoRecord" = "黑人+无前科",
  "race_recordHispanic_HasRecord" = "拉丁裔+有前科",
  "race_recordHispanic_NoRecord" = "拉丁裔+无前科",
  "race_recordOther_HasRecord" = "其他种族+有前科",
  "race_recordOther_NoRecord" = "其他种族+无前科",
  "race_recordWhite_HasRecord" = "白人+有前科",
  "race_recordWhite_NoRecord" = "白人+无前科"
)

# 添加中文特征名
var_imp_df$Feature_CN <- feature_names[var_imp_df$Feature]
var_imp_df$Feature_CN <- ifelse(is.na(var_imp_df$Feature_CN), 
                                var_imp_df$Feature, 
                                var_imp_df$Feature_CN)

# 绘制特征重要性图
ggplot(var_imp_df, aes(x = reorder(Feature_CN, Importance), y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "改进模型特征重要性排名",
       x = "特征",
       y = "重要性分数") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5))

ggsave("improved_feature_importance.png", width = 8, height = 6)
# PR曲线（比ROC更适合不平衡数据）
pr_curve <- pr.curve(
  scores.class0 = cv_predictions$Penitent,
  weights.class0 = as.numeric(cv_predictions$obs == "Penitent"),
  curve = TRUE
)

plot(pr_curve, main = "PR曲线 (AUC = {round(pr_curve$auc.integral, 3)})")

# 7. 改进预测
# 首先从训练数据获取因子水平
race_levels <- levels(train_data$race_simplified)
prior_levels <- levels(train_data$prior_record_factor)
race_record_levels <- levels(train_data$race_record)

predict_data <- combined_data %>% 
  filter(is.na(penitence)) %>%
  select(execution, education_years, race_simplified, 
         prior_record_factor, occupation_count) %>%
  mutate(
    # 处理种族特征
    race_simplified = factor(
      coalesce(as.character(race_simplified), "Other"),
      levels = race_levels
    ),
    
    # 处理前科记录特征
    prior_record_factor = factor(
      coalesce(as.character(prior_record_factor), "NoRecord"),
      levels = prior_levels
    ),
    
    # 创建race_record特征
    race_record = factor(
      paste(race_simplified, prior_record_factor, sep = "_"),
      levels = race_record_levels
    ),
    
    # 处理数值特征
    education_years = replace_na(education_years, 0),
    occupation_count = replace_na(occupation_count, 0),
    
    # 计算交互特征
    edu_occupation = education_years * occupation_count
  )

# 检查是否有新水平
new_levels <- setdiff(levels(predict_data$race_record), race_record_levels)
if (length(new_levels) > 0) {
  warning("发现新的race_record水平: ", paste(new_levels, collapse = ", "))
  
  # 获取训练数据中最常见的水平
  most_common <- names(sort(table(train_data$race_record), decreasing = TRUE))[1]
  
  # 将新水平映射到最常见的水平
  predict_data$race_record <- as.character(predict_data$race_record)
  predict_data$race_record[predict_data$race_record %in% new_levels] <- most_common
  predict_data$race_record <- factor(predict_data$race_record, levels = race_record_levels)
}

# 8. 模型性能验证
final_conf <- confusionMatrix(
  data = factor(ifelse(cv_predictions$Penitent > best_threshold, "Penitent", "NotPenitent"),
                levels = c("NotPenitent", "Penitent")),
  reference = cv_predictions$obs,
  positive = "Penitent"
)

print(final_conf)
cat(sprintf("\n改进模型F1分数: %.4f", final_conf$byClass["F1"]))


# 重新创建预测数据集（修复因子转换问题）
# 获取训练数据的因子水平
race_levels <- levels(train_data$race_simplified)
prior_levels <- levels(train_data$prior_record_factor)
race_record_levels <- levels(train_data$race_record)

predict_data <- combined_data %>% 
  filter(is.na(penitence)) %>%
  select(execution, education_years, race_simplified, 
         prior_record_factor, occupation_count) %>%
  mutate(
    # 处理数值型特征
    education_years = replace_na(education_years, 0),
    occupation_count = replace_na(occupation_count, 0),
    
    # 处理种族特征 - 分步转换
    race_char = as.character(race_simplified),
    race_char = ifelse(is.na(race_char), "Other", race_char),
    race_simplified = factor(race_char, levels = race_levels),
    
    # 处理前科记录特征
    prior_char = as.character(prior_record_factor),
    prior_char = ifelse(is.na(prior_char), "NoRecord", prior_char),
    prior_record_factor = factor(prior_char, levels = prior_levels),
    
    # 创建race_record特征
    race_record = paste(race_simplified, prior_record_factor, sep = "_"),
    race_record = factor(race_record, levels = race_record_levels),
    
    # 计算交互特征
    edu_occupation = education_years * occupation_count
  ) %>%
  select(-race_char, -prior_char)  # 移除临时列

# 检查预测数据
cat("预测数据集样本数:", nrow(predict_data), "\n")
if (nrow(predict_data) == 0) {
  stop("预测数据集为空！请检查数据预处理步骤。")
}

# 9. 预测集处理与结果保存
# 使用最佳阈值进行预测
predict_probs <- predict(model, newdata = predict_data, type = "prob")$Penitent
predict_class <- ifelse(predict_probs > best_threshold, "Penitent", "NotPenitent")

# 创建完整预测结果数据集
predictions_df <- predict_data %>%
  mutate(
    penitence_prob = predict_probs,
    predicted_class = factor(predict_class, levels = c("NotPenitent", "Penitent"))
  ) %>%
  select(execution, education_years, race_simplified, 
         prior_record_factor, occupation_count, 
         edu_occupation, race_record,
         penitence_prob, predicted_class)

# 保存预测结果
write_csv(predictions_df, "predictions_results.csv")
cat("\n预测结果已保存至: predictions_results.csv\n")

# 10. 预测结果可视化
# 预测类别分布
class_distribution <- predictions_df %>%
  count(predicted_class) %>%
  mutate(percentage = n / sum(n) * 100)

p1 <- ggplot(class_distribution, aes(x = predicted_class, y = n, fill = predicted_class)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = paste0(n, "\n(", round(percentage, 1), "%)")), 
            vjust = -0.3, size = 4) +
  scale_fill_manual(values = c("NotPenitent" = "#F8766D", "Penitent" = "#00BFC4"),
                    labels = c("NotPenitent" = "不忏悔", "Penitent" = "忏悔")) +
  labs(title = "预测类别分布",
       x = "忏悔状态",
       y = "样本数量") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "none")

ggsave("predicted_class_distribution.png", p1, width = 7, height = 6)

# 将混淆矩阵转换为数据框
conf_matrix_df <- as.data.frame(final_conf$table)
colnames(conf_matrix_df) <- c("Predicted", "Actual", "Freq")

# 添加中文标签
conf_matrix_df <- conf_matrix_df %>%
  mutate(
    Actual_CN = case_when(
      Actual == "Penitent" ~ "实际:忏悔",
      Actual == "NotPenitent" ~ "实际:不忏悔"
    ),
    Predicted_CN = case_when(
      Predicted == "Penitent" ~ "预测:忏悔",
      Predicted == "NotPenitent" ~ "预测:不忏悔"
    )
  )

# 计算百分比
conf_matrix_df <- conf_matrix_df %>%
  group_by(Actual) %>%
  mutate(Percentage = Freq / sum(Freq) * 100) %>%
  ungroup()

# 创建热力图
conf_plot <- ggplot(conf_matrix_df, aes(x = Predicted_CN, y = Actual_CN, fill = Freq)) +
  geom_tile(color = "white", alpha = 0.8) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", Freq, Percentage)), 
            color = "black", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "#F0F8FF", high = "#4682B4", name = "样本数量") +
  labs(title = "混淆矩阵 - 模型性能评估",
       subtitle = sprintf("准确率: %.2f%%, F1分数: %.3f", 
                          final_conf$overall["Accuracy"] * 100,
                          final_conf$byClass["F1"]),
       x = "预测类别",
       y = "实际类别") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    legend.position = "right"
  ) +
  coord_fixed()

# 保存混淆矩阵图
ggsave("confusion_matrix.png", conf_plot, width = 8, height = 6)
cat("\n混淆矩阵图已保存至: confusion_matrix.png\n")




# 忏悔概率分布
p2 <- ggplot(predictions_df, aes(x = penitence_prob)) +
  geom_histogram(binwidth = 0.05, fill = "#619CFF", alpha = 0.8, color = "white") +
  geom_vline(xintercept = best_threshold, linetype = "dashed", color = "red", size = 1) +
  annotate("text", x = best_threshold + 0.05, y = Inf, 
           label = paste("阈值 =", round(best_threshold, 2)), 
           vjust = 1.5, color = "red", size = 5) +
  labs(title = "忏悔概率分布",
       x = "忏悔概率",
       y = "样本数量") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggsave("penitence_prob_distribution.png", p2, width = 8, height = 6)

# 按种族和前科记录分组分析
if("race_simplified" %in% names(predictions_df) && "prior_record_factor" %in% names(predictions_df)) {
  grouped_prediction <- predictions_df %>%
    group_by(race_simplified, prior_record_factor, predicted_class) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(race_simplified, prior_record_factor) %>%
    mutate(group_total = sum(n),
           percentage = n / group_total * 100) %>%
    ungroup()
  
  # 创建中文标签
  race_labels <- c("White" = "白人", "Black" = "黑人", 
                   "Hispanic" = "拉丁裔", "Other" = "其他")
  record_labels <- c("NoRecord" = "无前科", "HasRecord" = "有前科")
  
  p3 <- ggplot(grouped_prediction, 
               aes(x = interaction(race_simplified, prior_record_factor, sep = " + "), 
                   y = percentage, fill = predicted_class)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.9) +
    geom_text(aes(label = paste0(round(percentage, 1), "%")), 
              position = position_dodge(width = 0.9), 
              vjust = -0.3, size = 3.5) +
    scale_fill_manual(values = c("NotPenitent" = "#F8766D", "Penitent" = "#00BFC4"),
                      labels = c("NotPenitent" = "不忏悔", "Penitent" = "忏悔")) +
    scale_x_discrete(labels = function(x) {
      parts <- strsplit(as.character(x), "\\.")
      sapply(parts, function(p) paste(race_labels[p[1]], record_labels[p[2]], sep = " + "))
    }) +
    labs(title = "按种族和前科记录的预测分布",
         x = "种族 + 前科记录",
         y = "百分比 (%)",
         fill = "预测类别") +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 10))
  
  ggsave("prediction_by_race_record.png", p3, width = 10, height = 7)
}

# 教育年限与忏悔概率关系
if("education_years" %in% names(predictions_df)) {
  p4 <- ggplot(predictions_df, aes(x = education_years, y = penitence_prob)) +
    geom_jitter(alpha = 0.6, color = "#619CFF", size = 2.5, width = 0.2) +
    geom_smooth(method = "loess", se = TRUE, color = "#F8766D", fill = "lightpink") +
    labs(title = "教育年限与忏悔概率关系",
         x = "教育年限",
         y = "忏悔概率") +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  # 职业数量与忏悔概率关系
  p5 <- ggplot(predictions_df, aes(x = occupation_count, y = penitence_prob)) +
    geom_jitter(alpha = 0.6, color = "#00BA38", size = 2.5, width = 0.2) +
    geom_smooth(method = "loess", se = TRUE, color = "#F8766D", fill = "lightpink") +
    labs(title = "职业数量与忏悔概率关系",
         x = "职业数量",
         y = "忏悔概率") +
    theme_minimal(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  # 合并教育和工作图表
  install.packages(patchwork)
  library(patchwork)
  combined_plot <- p4 + p5 + plot_layout(ncol = 2)
  ggsave("edu_occ_relationship.png", combined_plot, width = 14, height = 6)
}

# 11. 生成模型报告
sink("model_report.txt")
cat("========== 最终模型报告 ==========\n\n")
cat("模型类型: 随机森林\n")
cat("最佳分类阈值:", round(best_threshold, 3), "\n\n")

cat("训练集类别分布:\n")
print(table(train_data$penitence_binary))
cat("\n")

cat("预测集类别分布:\n")
print(table(predictions_df$predicted_class))
cat("\n")

cat("模型性能指标 (交叉验证):\n")
print(final_conf$byClass[c("Sensitivity", "Specificity", "Precision", "Recall", "F1")])
cat("\n")

cat("特征重要性排名 (Top 10):\n")
top_features <- var_imp_df %>%
  arrange(desc(Importance)) %>%
  head(10) %>%
  select(Feature_CN, Importance)
print(top_features)
cat("\n")

cat("预测结果已保存至: predictions_results.csv\n")
cat("可视化图表已保存至当前目录\n")
sink()

cat("\n========== 分析完成! ==========\n")
cat("生成报告: model_report.txt\n")

