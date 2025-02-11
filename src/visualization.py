"""
Visualization Functions for plotting Freezing data, Moseq data, and other data
"""


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def plot_freezing_time(data, variable, effect_size, pvalue, title_text, hue, ylim=[0, 60], output_filename='Freezing Time RM-ANOVA.svg'):
    """
    Plots the preferred subsets of freezing time data for sefla stage and other stages side by side using a double axis plot. 

    Parameters:
    data (pd.DataFrame): A DataFrame containing the freezing time data with a 'day' column and 'freezing' column.
    effect_size (float): The effect size of the difference between the two groups of data using repeated-measure ANOVA. 
    pvalue (float): The p-value of the difference between the two groups of data using repeated-measure ANOVA.
    title_text (str): The title text to be displayed on the plot specifying the data being compared.
    hue (str): The column name used for grouping data (e.g., treatment group).
    ylim (list): The y-axis limits for the plot. Default is [0, 60].
    output_filename (str): The filename to save the plot as. Default is 'Freezing Time RM-ANOVA.svg'.

    Returns:
    A double axis plot comparing the freezing time data between groups for the sefla stage and other stages side by side.
    """
    # Subset the data
    sefla_data = data[data['day'] == 'sefla']
    subset_data = data[data['day'].isin(['seflb', 'recall1', 'recall2', 'recall3', 'recall4'])]

    plt.rcParams['font.family'] = 'Arial'  # Choose a vector-safe font

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True, gridspec_kw={'width_ratios': [1, 4]})

    # Plot the sefla data on the first axis
    sns.pointplot(ax=ax1, data=sefla_data, x='day', y=variable, hue=hue, join=True)
    ax1.set_ylabel('Freezing Time (sec)')
    ax1.set_ylim(ylim)
    ax1.set_xlabel('')
    ax1.legend().remove()
    sns.despine(ax=ax1)

    # Plot the non-sefla data on the second axis
    sns.pointplot(ax=ax2, data=subset_data, x='day', y=variable, hue=hue, join=True)
    ax2.set_xlabel('')
    ax2.set_ylabel('Freezing Time (sec)')
    ax2.legend(title=hue)
    sns.despine(ax=ax2)

    # Add main title
    fig.suptitle(f'{title_text}', fontsize=16, x=0.5)
    fig.text(0.5, -0.05, 'Experimental Stage', ha='center', fontsize=12)

    # Determine comparison sign for p-value
    pvalue_sign = '<' if pvalue < 0.05 else '>'

    # Add effect size and p-value annotation
    plt.text(0.5, 0.9, f"Effect size: {effect_size:.2f}", transform=plt.gca().transAxes)
    plt.text(0.5, 0.85, f"p-value: {pvalue:.2e} {pvalue_sign} .05", transform=plt.gca().transAxes)

    # Adjust the layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make space for the main title
    plt.savefig(output_filename, format='svg')
    plt.show()




def create_violin_plot(data, x, y, hue, title, ylabel, significant_syllables):
    """
    plots the significant syllables in a violin plot

    Parameters:
    data (pd.DataFrame): The data to plot (e.g., heading mean, angular velocity, duration, etc.)
    x (str): The x-axis variable to plot (e.g., syllable index)
    y (str): The y-axis variable to plot (e.g., angular velocity)
    hue (str): The hue variable to plot (e.g., sefl vs. control, young vs. old)
    title (str): The title of the plot
    ylabel (str): The label of the y-axis (e.g., 'Angular Velocity (deg/s)')
    significant_syllables (list): The list of significant syllables to plot

    """
    data = data[data['syllable'].isin(significant_syllables)]
    if data.empty:
        return print('No significant syllables to plot')
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(data=data, x=x, y=y, hue=hue, split=True, inner='quartile')
    plt.title(title)
    plt.xlabel('Syllable Index')
    plt.ylabel(ylabel)

    
    plt.show()


def create_box_strip_plot(data, x, y, hue, title, ylabel, significant_syllables, ylim=None):
    """
    plots the significant syllables in a box plot

    Parameters:
    data (pd.DataFrame): The data to plot (e.g., heading mean, angular velocity, duration, etc.)
    x (str): The x-axis variable to plot (e.g., syllable index)
    y (str): The y-axis variable to plot (e.g., angular velocity)
    hue (str): The hue variable to plot (e.g., sefl vs. control, young vs. old)
    title (str): The title of the plot
    ylabel (str): The label of the y-axis (e.g., 'Angular Velocity (deg/s)')
    significant_syllables (list): The list of significant syllables to plot

    """
    data = data[data['syllable'].isin(significant_syllables)]
    if data.empty:
        return print('No significant syllables to plot')

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=x, y=y, hue=hue, showfliers=False)
    
    plt.title(title)
    plt.xlabel('Syllable Index')
    plt.ylabel(ylabel)
    
    plt.show()