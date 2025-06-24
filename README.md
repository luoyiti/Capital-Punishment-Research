# Capital-Punishment-Research
a data analysis program based on capital punishment datasets.

# 总体架构

- data
  - 存有美国死刑执行记录数据集
    - capital-punishment-to-2024.csv
    - executions-to-2002.csv（本研究主要使用这个）
  - 经过独热编码后的死刑执行记录数据集
    - all_encoded.csv
  - 经过人工打分处理后的死刑犯遗言数据
    - last_words_sampled_rows.csv
  - 死刑犯遗言数据汇总
    - final_data_setv1.xlsx
    - raw_last_statement.csv
  - twitter死刑话题讨论数据
    - twitter_capital_data.csv
    - capital.h5
  - 其它
    - 停用词：stop_words.txt
- data-preprocessing
  - 抽取死刑遗言记录
    - last_word_random_pick.R
  - 爬取死刑遗言数据
    - last_words_crawler.ipynb
  - 爬取处理推特死刑话题数据
    - twitter_data_extract.ipynb
    - twitter_data_info.ipynb
- data-analysize
  - 死刑记录数据描述性统计
    - Analysis of potential influencing factors.R
    - location_visualize.py
  - 聚类分析
    - cluster-analysize
  - 假设检验
    - Analysis of potential influencing factors.R
  - 对死刑犯遗言数据的机器学习分析与总体分析
    - decision_tree_model.ipynb
    - tree.py
    - machine_learningand_descriptive_analysize_for_capital_penalty.ipynb
    - NLP_For_Last_Words_and_last_statements_analysize.ipynb
  - 死刑话题数据分析
    - descriptive_statistics.py
  - 时间序列分析
    - time_series_analysis.ipynb
- visualize
  - 包括所有可视化分析后图片的存储
- 附录文件
  - 初步提案
  - 中期汇报
  - 期末报告
  - 期末汇报ppt
  - 项目思维导图
  - 项目流程图
  - 死刑犯遗言评判要点

# 研究简介

**核心问题**：

死刑制度的内在建制和外在影响机制是什么？

**研究方法**：

本研究将基于美国建国以来的死刑执行与冤案记录、德州死刑犯遗言数据以及Twitter上与死刑话题相关的舆情数据，采用统计检验、机器学习与自然语言处理等方法，进行死刑制度社会歧视性分析、死刑犯悔过程度探究和死刑执行的群体态度探究，分别从死刑制度的内在属性和外在影响,探索其意义和价值。

**核心目的**：

从制度机制与社会反应双重视角审视死刑制度的价值与问题，为司法审慎适用死刑、推动制度改革与法律正义提供科学证据支持，促使社会在尊重生命价值基础上达成更具共识性的刑罚判断。
