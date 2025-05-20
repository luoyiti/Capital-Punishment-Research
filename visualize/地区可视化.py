import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.globals import ThemeType

# 1. 数据加载与预处理
df = pd.read_csv("C:/Users/tpj/Desktop/executions-to-2002.csv")

# 检查原始数据中各州分布情况
print("原始数据中各州分布：")
print(df['STATE OF CONVICTION'].value_counts())

# 2. 按地区分类统计
# 提取州名称
state_pattern = r'\((\d+)\)\s+([^(]+)'  # 匹配括号内数字和后面的州名
df['州名称'] = df['STATE OF CONVICTION'].str.extract(state_pattern)[1]

# 定义美国州全称到缩写的映射
state_full_to_abbr = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# 映射州名称到缩写，并处理可能的缺失值
df['州缩写'] = df['州名称'].map(state_full_to_abbr)

# 过滤掉无法映射的州
valid_data = df.dropna(subset=['州缩写'])

# 统计各州执行次数
state_execution_counts = valid_data['州缩写'].value_counts().reset_index()
state_execution_counts.columns = ['州缩写', '执行次数']

# 打印统计摘要
print("\n各州执行次数统计摘要：")
print(f"最大值: {state_execution_counts['执行次数'].max()}")
print(f"最小值: {state_execution_counts['执行次数'].min()}")
print(f"平均值: {state_execution_counts['执行次数'].mean():.2f}")
print(f"中位数: {state_execution_counts['执行次数'].median()}")

# 定义美国州缩写到全称的映射
state_abbr_to_full = {v: k for k, v in state_full_to_abbr.items()}

# 将州缩写转换为英文全称
state_execution_counts['州全称'] = state_execution_counts['州缩写'].map(state_abbr_to_full)

# 准备地图数据
map_data = [tuple(x) for x in state_execution_counts[['州全称', '执行次数']].values]

# 自定义分段区间，使用红色系颜色
pieces = [
    {"min": 0, "max": 50, "label": "0 - 50次", "color": "#FFEBEB"},
    {"min": 51, "max": 100, "label": "51 - 100次", "color": "#FFC2C2"},
    {"min": 101, "max": 300, "label": "101 - 300次", "color": "#FF8989"},
    {"min": 301, "max": 800, "label": "301 - 800次", "color": "#FF5050"},
    {"min": 801, "label": "800次以上", "color": "#FF0000"}
]

# 创建地图图表
c2 = (
    Map(init_opts=opts.InitOpts(
        theme=ThemeType.LIGHT,
        width="1200px",
        height="800px"
    ))
    .add(
        series_name="执行次数",
        data_pair=map_data,
        maptype="美国",
        is_map_symbol_show=False,
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="美国各州死刑执行次数分布 (1812 - 2002)",
            subtitle="数据来源：executions - to - 2002.csv",
            pos_left="center"
        ),
        visualmap_opts=opts.VisualMapOpts(
            is_piecewise=True,
            pieces=pieces,
            orient="vertical",
            pos_left="right",
            textstyle_opts=opts.TextStyleOpts(color="#000", font_size=14)
        ),
        toolbox_opts=opts.ToolboxOpts(is_show=True)
    )
)

# 保存地图
c2.render("usa_execution_distribution_piecewise.html")

print("\n地图已生成：")
print("usa_execution_distribution_piecewise.html (分段颜色映射)")
print("请在浏览器中打开查看交互式地图")