import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2
bmi = df['weight'] / ((df['height']/100)**2)
df['overweight'] = (bmi > 25).astype(int)

# 3
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc'] = 1

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'],
                    value_name='value', var_name='feature')


    # 7
    graph = sns.catplot(data=df_cat, x='feature', hue='value', col='cardio', kind='count', height=5, aspect=1.5)
    graph.set_axis_labels('variable', 'total')

    # 8
    fig = graph.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df

    # Clean the data
    df_heat = df_heat[
        (df_heat['ap_lo'] <= df_heat['ap_hi']) &
        (df_heat['height'] >= df_heat['height'].quantile(0.025)) &
        (df_heat['height'] <= df_heat['height'].quantile(0.975)) &
        (df_heat['weight'] >= df_heat['weight'].quantile(0.025)) &
        (df_heat['weight'] <= df_heat['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr().round(1)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))


    # 14
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 15
    sns.heatmap(
        corr,
        cmap='RdBu',
        mask=mask,
        fmt='.1f',
        annot=True,
        square=True
    )


    # 16
    fig.savefig('heatmap.png')
    return fig

draw_heat_map()