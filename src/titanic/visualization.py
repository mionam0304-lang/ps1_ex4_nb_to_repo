import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style():
    """
    Unify drawing style
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette(["#A5D7E8", "#576CBC", "#19376D", "#0b2447"])

def plot_count_pairs(df, column: str, hue_column: str = None, title: str = None, figsize: tuple = (8, 4)):
    """
    Draw a count pair graph
    """
    set_plot_style()
    f, ax = plt.subplots(1, 1, figsize=figsize)
    
    if hue_column:
        sns.countplot(x=column, data=df, hue=hue_column, palette=["#A5D7E8", "#576CBC"])
    else:
        sns.countplot(x=column, data=df, palette=["#A5D7E8", "#576CBC"])
    
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Number of passengers / {column}")
    
    plt.show()

def plot_distribution_pairs(df, column: str, hue_column: str = None, title: str = None, figsize: tuple = (8, 4)):
    """
    Draw a distribution pair graph
    """
    set_plot_style()
    f, ax = plt.subplots(1, 1, figsize=figsize)
    
    color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
    
    if hue_column:
        for i, h in enumerate(df[hue_column].unique()):
            sns.histplot(df.loc[df[hue_column] == h, column], 
                        color=color_list[i % len(color_list)], 
                        ax=ax, 
                        label=h)
        ax.legend()
    else:
        sns.histplot(df[column], color=color_list[0], ax=ax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Distribution of {column}")
    
    plt.show()