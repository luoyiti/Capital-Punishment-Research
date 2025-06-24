# 加载必要库
library(tidyverse)
library(lubridate)

# 读取数据
df <- read_csv("美国1972年后死刑犯.csv")

# 清洗州名（Jurisdiction列可能存在不一致的命名）
df <- df %>%
  mutate(Jurisdiction = str_to_title(Jurisdiction)) %>%  # 统一州名格式
  filter(!is.na(Jurisdiction), !is.na(Year))  # 去除缺失值

# 查看唯一州列表
unique_states <- unique(df$Jurisdiction)
print(unique_states)

# 创建政治倾向数据（示例）
#political_affiliation <- tibble(
#  Jurisdiction = c("Texas", "California", "Florida", "Pennsylvania", "Ohio"),
#  Political_Party = c("Republican", "Democratic", "Republican", "Democratic", "Republican")
#)

# 定义所有州及其政治倾向（2020年大选结果 + 历史趋势）
political_affiliation <- tibble::tribble(
  ~Jurisdiction,          ~Political_Party,
  #------------------------|-----------------
  "Alabama",              "Republican",
  "Alaska",               "Republican",
  "Arizona",              "Democratic",  # 2020年翻蓝
  "Arkansas",             "Republican",
  "California",           "Democratic",
  "Colorado",             "Democratic",
  "Connecticut",          "Democratic",
  "Delaware",             "Democratic",
  "Florida",              "Republican",
  "Georgia",              "Democratic",  # 2020年翻蓝
  "Hawaii",               "Democratic",
  "Idaho",                "Republican",
  "Illinois",             "Democratic",
  "Indiana",              "Republican",
  "Iowa",                 "Republican",
  "Kansas",               "Republican",
  "Kentucky",             "Republican",
  "Louisiana",            "Republican",
  "Maine",                "Democratic",
  "Maryland",             "Democratic",
  "Massachusetts",        "Democratic",
  "Michigan",             "Democratic",
  "Minnesota",            "Democratic",
  "Mississippi",          "Republican",
  "Missouri",             "Republican",
  "Montana",              "Republican",
  "Nebraska",             "Republican",
  "Nevada",               "Democratic",
  "New Hampshire",        "Democratic",
  "New Jersey",           "Democratic",
  "New Mexico",           "Democratic",
  "New York",             "Democratic",
  "North Carolina",       "Republican",
  "North Dakota",         "Republican",
  "Ohio",                 "Republican",
  "Oklahoma",             "Republican",
  "Oregon",               "Democratic",
  "Pennsylvania",         "Democratic",  # 2020年翻蓝
  "Rhode Island",         "Democratic",
  "South Carolina",       "Republican",
  "South Dakota",         "Republican",
  "Tennessee",            "Republican",
  "Texas",                "Republican",
  "Utah",                 "Republican",
  "Vermont",              "Democratic",
  "Virginia",             "Democratic",
  "Washington",           "Democratic",
  "West Virginia",        "Republican",
  "Wisconsin",            "Democratic",
  "Wyoming",              "Republican",
  "District of Columbia", "Democratic"   # 华盛顿特区
)

# 检查是否有遗漏的州
all_states <- datasets::state.name
setdiff(all_states, political_affiliation$Jurisdiction)  # 应返回空

# 合并到主数据集
df <- df %>%
  left_join(political_affiliation, by = "Jurisdiction")

# 定义关键时间节点
df <- df %>%
  mutate(
    Period = case_when(
      Year < 1972 ~ "Pre-Furman (Before 1972)",
      Year >= 1972 & Year < 1976 ~ "Furman Era (1972-1976)",
      Year >= 1976 ~ "Post-Gregg (After 1976)"
    )
  )

# 按州和政治倾向统计死刑数量
state_summary <- df %>%
  group_by(Jurisdiction, Political_Party) %>%
  summarise(Execution_Count = n(), .groups = "drop")

# 按政治党派统计
party_summary <- df %>%
  group_by(Political_Party) %>%
  summarise(Execution_Count = n(), .groups = "drop")

# 按时间段统计
period_summary <- df %>%
  group_by(Period) %>%
  summarise(Execution_Count = n(), .groups = "drop")

print(state_summary)
print(party_summary)
print(period_summary)

# 政治党派与死刑数量
ggplot(party_summary, aes(x = Political_Party, y = Execution_Count, fill = Political_Party)) +
  geom_bar(stat = "identity") +
  labs(title = "Executions by Political Affiliation of State", x = "Party", y = "Count")

# 时间段趋势
ggplot(period_summary, aes(x = Period, y = Execution_Count, fill = Period)) +
  geom_bar(stat = "identity") +
  labs(title = "Executions by Historical Period", x = "Period", y = "Count")

# 州级分布（示例：前10州）
top_states <- state_summary %>%
  arrange(desc(Execution_Count)) %>%
  head(10)

ggplot(top_states, aes(x = reorder(Jurisdiction, Execution_Count), y = Execution_Count, fill = Political_Party)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top 10 States by Execution Count", x = "State", y = "Count")

# 卡方检验（党派与死刑执行）
chisq_test <- chisq.test(table(df$Political_Party))
print(chisq_test)



