import pandas as pd
import numpy as np

def create_latex_table_simulation(df, caption="Comparison of NCA variants", label="tab:nca-comparison"):
    """Convert results DataFrame to a LaTeX table following ICML format"""
    # Create copy to avoid modifying original
    df_latex = df.copy()
    
    def extract_value(s):
        return float(s.split('±')[0].strip())
    
    # Select metrics to include
    metrics = ['KL Divergence', 'Chi-Square', 'Tumor Size Diff', 'Border Size Diff']
    
    # Process metrics
    for metric in metrics:
        df_latex[metric] = df_latex[metric] + ' ' + df_latex[f'{metric} SD']
        values = df_latex[metric].apply(extract_value)
        best_idx = values.idxmin()
        df_latex.loc[best_idx, metric] = '\\textbf{' + df_latex.loc[best_idx, metric] + '}'
        df_latex = df_latex.drop(f'{metric} SD', axis=1)
    
    # Rename columns
    column_renames = {
        'Model Type': 'Model',
        'KL Divergence': 'KL-div',
        'Chi-Square': '$\\chi^2$',
        'Tumor Size Diff': 'Size',
        'Border Size Diff': 'Border-W'
    }
    df_latex = df_latex.rename(columns=column_renames)
    
    # Shorten model names
    model_renames = {
        'Standard NCA': 'NCA',
        'Mixture NCA': 'Mix-NCA',
        'Stochastic Mixture NCA': 'SMix-NCA'
    }
    df_latex['Model'] = df_latex['Model'].replace(model_renames)
    
    # Keep only selected columns
    df_latex = df_latex[['Model', 'KL-div', '$\\chi^2$', 'Size', 'Border-W']]
    
    # Build table string parts
    table_header = (
        "\\begin{table}[t]\n"
        "\\caption{" + caption + "}\n"
        "\\label{" + label + "}\n"
        "\\vskip 0.15in\n"
        "\\begin{center}\n"
        "\\begin{small}\n"
        "\\begin{sc}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Model & KL-div & $\\chi^2$ & Size & Border-W \\\\\n"
        "\\midrule\n"
    )
    
    # Add rows
    table_body = ""
    for _, row in df_latex.iterrows():
        table_body += "{} & {} & {} & {} & {} \\\\\n".format(
            row['Model'], 
            row['KL-div'], 
            row['$\\chi^2$'], 
            row['Size'], 
            row['Border-W']
        )
    
    # Table footer
    table_footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{sc}\n"
        "\\end{small}\n"
        "\\end{center}\n"
        "\\vskip -0.1in\n"
        "\\end{table}\n"
    )
    
    # Combine all parts
    latex_code = table_header + table_body + table_footer
    
    return latex_code


def create_latex_table_emojis(df, final_losses):
    # Create emoji mapping using twemoji package commands
    emoji_map = {
        '1F914': '\\twemoji{thinking}',
        '1F620': '\\twemoji{angry}',
        '1F604': '\\twemoji{smile}',
        '1F951': '\\twemoji{avocado}',
        '1F433': '\\twemoji{whale}',
        '1F984': '\\twemoji{unicorn}'
    }
    
    # Rename model types
    model_rename = {
        'Standard': 'NCA',
        'Mixture': 'MNCA',
        'Stochastic': 'MNCA w/ Noise'
    }
    
    # Create a copy of df with renamed models
    df_renamed = df.copy()
    df_renamed['Model Type'] = df_renamed['Model Type'].replace(model_rename)
    
    # Group and compute statistics
    summary = df_renamed.groupby(['emoji', 'Model Type', 'Perturbation Type'])['Final Error'].agg(['mean', 'std']).round(4)
    summary_table = summary.unstack()
    
    def format_cell(mean, std):
        return f"{mean:.3f} ±{std:.3f}"
    
    # Format the table
    formatted_table = pd.DataFrame(
        index=summary_table.index,
        columns=summary_table.columns.levels[1],
        data=[[format_cell(summary_table[('mean', pt)][idx], summary_table[('std', pt)][idx]) 
               for pt in summary_table.columns.levels[1]]
              for idx in summary_table.index]
    )
    
    # Sort the index to ensure consistent model order
    model_order = ['NCA', 'MNCA', 'MNCA w/ Noise']
    formatted_table = formatted_table.reindex(
        [(e, m) for e in formatted_table.index.get_level_values(0).unique() 
         for m in model_order]
    )
    
    def get_mean(value_str):
        return float(value_str.split('±')[0].strip())
    
    # Convert to LaTeX
    latex_table = (
        "\\begin{table}[t]\n"
        "\\caption{Performance Across Different Perturbations}\n"
        "\\label{tab:perturbation-results}\n"
        "\\begin{center}\n"
        "\\begin{small}\n"
        "\\begin{sc}\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n"
        "\\toprule\n"
        "Model & Del 5x5px & Del 10x10px & Noise 10\\% & Noise 25\\% & Removal 100px & Removal 500px \\\\\n"
        "\\midrule\n"
    )
    
    current_emoji = None
    for (emoji_code, model_type) in formatted_table.index:
        if current_emoji != emoji_code:
            if current_emoji is not None:
                latex_table += "\\midrule\n"
            current_emoji = emoji_code
            latex_table += f"\\multicolumn{{7}}{{l}}{{{emoji_map[emoji_code]}}} \\\\\n"
        
        # Get values for current emoji and find best
        emoji_data = formatted_table.loc[formatted_table.index.get_level_values(0) == emoji_code]
        
        # Get final loss
        if emoji_code in final_losses:
            model_key = {
                'NCA': 'standard',
                'MNCA': 'mixture',
                'MNCA w/ Noise': 'stochastic'
            }[model_type]
            final_loss = final_losses[emoji_code][f'{model_key}_final_loss']
        else:
            final_loss = 0.0
        
        # Format row values with bold for best performance
        row_values = [model_type]  # Already renamed
        
        # Add values for each perturbation type
        for col in ['Deletion 5', 'Deletion 10', 'Noise 0.1', 'Noise 0.25', '100 Masked Pixels', '500 Masked Pixels']:
            values = [get_mean(x) for x in emoji_data[col]]
            best_value = min(values)
            current_value = get_mean(formatted_table.loc[(emoji_code, model_type), col])
            if abs(current_value - best_value) < 1e-6:
                row_values.append('\\textbf{' + formatted_table.loc[(emoji_code, model_type), col] + '}')
            else:
                row_values.append(formatted_table.loc[(emoji_code, model_type), col])
        
        # Add final loss with bold for best
        losses = [final_losses[emoji_code][f'{m}_final_loss'] 
                 for m in ['standard', 'mixture', 'stochastic']]
        best_loss = min(losses)
        if abs(final_loss - best_loss) < 1e-6:
            row_values.append(f'\\textbf{{{final_loss:.3f}}}')
        else:
            row_values.append(f'{final_loss:.3f}')
        
        latex_table += " & ".join(row_values) + " \\\\\n"
    
    latex_table += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{sc}\n"
        "\\end{small}\n"
        "\\end{center}\n"
        "\\vskip -0.1in\n"
        "\\end{table}\n"
    )
    
    return latex_table


def create_latex_table_emojis_rebuttal(df):
    # Create emoji mapping using twemoji package commands
    emoji_map = {
        '1F914': '\\twemoji{thinking}',
        '1F620': '\\twemoji{angry}',
        '1F604': '\\twemoji{smile}',
        '1F951': '\\twemoji{avocado}',
        '1F433': '\\twemoji{whale}',
        '1F984': '\\twemoji{unicorn}'
    }
    
    # Rename model types
    model_rename = {
        'Standard': 'NCA',
        'Mixture': 'MNCA',
        'Stochastic': 'MNCA w/ Noise', 
        'GCA': 'GCA'
    }
    
    # Create a copy of df with renamed models
    df_renamed = df.copy()
    df_renamed['Model Type'] = df_renamed['Model Type'].replace(model_rename)
    
    # Group and compute statistics
    summary = df_renamed.groupby(['emoji', 'Model Type', 'Perturbation Type'])['Final Error'].agg(['mean', 'std']).round(4)
    summary_table = summary.unstack()
    
    def format_cell(mean, std):
        return f"{mean:.3f} ±{std:.3f}"
    
    # Format the table
    formatted_table = pd.DataFrame(
        index=summary_table.index,
        columns=summary_table.columns.levels[1],
        data=[[format_cell(summary_table[('mean', pt)][idx], summary_table[('std', pt)][idx]) 
               for pt in summary_table.columns.levels[1]]
              for idx in summary_table.index]
    )
    
    # Sort the index to ensure consistent model order
    model_order = ['NCA', 'GCA','MNCA', 'MNCA w/ Noise']
    formatted_table = formatted_table.reindex(
        [(e, m) for e in formatted_table.index.get_level_values(0).unique() 
         for m in model_order]
    )
    
    def get_mean(value_str):
        return float(value_str.split('±')[0].strip())
    
    # Convert to LaTeX
    latex_table = (
        "\\begin{table}[t]\n"
        "\\caption{Performance Across Different Perturbations}\n"
        "\\label{tab:perturbation-results}\n"
        "\\begin{center}\n"
        "\\begin{small}\n"
        "\\begin{sc}\n"
        "\\begin{tabular}{@{}lcccccc@{}}\n"
        "\\toprule\n"
        "Model & Del 5x5px & Del 10x10px & Noise 10\\% & Noise 25\\% & Removal 100px & Removal 500px \\\\\n"
        "\\midrule\n"
    )
    
    current_emoji = None
    for (emoji_code, model_type) in formatted_table.index:
        if current_emoji != emoji_code:
            if current_emoji is not None:
                latex_table += "\\midrule\n"
            current_emoji = emoji_code
            latex_table += f"\\multicolumn{{7}}{{l}}{{{emoji_map[emoji_code]}}} \\\\\n"
        
        # Get values for current emoji and find best
        emoji_data = formatted_table.loc[formatted_table.index.get_level_values(0) == emoji_code]
        
        # Format row values with bold for best performance
        row_values = [model_type]  # Already renamed
        
        # Add values for each perturbation type
        for col in ['Deletion 5', 'Deletion 10', 'Noise 0.1', 'Noise 0.25', '100 Masked Pixels', '500 Masked Pixels']:
            values = [get_mean(x) for x in emoji_data[col]]
            best_value = min(values)
            current_value = get_mean(formatted_table.loc[(emoji_code, model_type), col])
            if abs(current_value - best_value) < 1e-6:
                row_values.append('\\textbf{' + formatted_table.loc[(emoji_code, model_type), col] + '}')
            else:
                row_values.append(formatted_table.loc[(emoji_code, model_type), col])
        
        
        latex_table += " & ".join(row_values) + " \\\\\n"
    
    latex_table += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{sc}\n"
        "\\end{small}\n"
        "\\end{center}\n"
        "\\vskip -0.1in\n"
        "\\end{table}\n"
    )
    
    return latex_table


def create_latex_table_cifar(df):
    """
    Create LaTeX table from DataFrame with CIFAR results
    Args:
        df: DataFrame with MultiIndex (Category, Model Type) and perturbation columns
    """
    # Start LaTeX table
    latex_table = (
        "\\begin{table}[t]\n"
        "\\caption{Performance Across Different Perturbations}\n"
        "\\label{tab:perturbation-results}\n"
        "\\begin{center}\n"
        "\\begin{small}\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "Model & Del 5x5px & Del 10x10px & Noise 10\\% & Noise 25\\% & Rem 100px & Rem 500px \\\\\n"
        "\\midrule\n"
    )
    
    # Get unique categories
    categories = df.index.get_level_values('Category').unique()
    
    # Column mapping
    col_map = {
        '100 Masked Pixels': 'Rem 100px',
        '500 Masked Pixels': 'Rem 500px',
        'Deletion 10': 'Del 10x10px',
        'Deletion 5': 'Del 5x5px',
        'Noise 0.1': 'Noise 10\\%',
        'Noise 0.25': 'Noise 25\\%'
    }
    
    # Process each category
    for i, category in enumerate(categories):
        # Add category header
        latex_table += f"\\multicolumn{{7}}{{l}}{{{category}}} \\\\\n"
        
        # Get data for this category
        category_data = df.loc[category]
        
        # Find minimum values for each column in this category
        min_values = category_data.apply(lambda x: float(x.str.split('±').str[0].astype(float).min()))
        
        # Process each model in the category
        for model_type in ['Standard', 'Mixture', 'Stochastic']:
            if model_type in category_data.index:
                line = [model_type.replace('Standard', 'NCA').replace('Mixture', 'MNCA').replace('Stochastic', 'MNCA w/ N')]
                
                # Add each metric
                for col in df.columns:
                    value = category_data.loc[model_type, col]
                    mean = float(value.split('±')[0])
                    std = float(value.split('±')[1])
                    
                    # Bold if this is the minimum value (within tolerance)
                    if np.isclose(mean, min_values[col], rtol=1e-5):
                        line.append(f"\\textbf{{{mean:.4f} ±{std:.4f}}}")
                    else:
                        line.append(f"{mean:.4f} ±{std:.4f}")
                
                latex_table += " & ".join(line) + " \\\\\n"
        
        # Add midrule between categories (except for last category)
        if i < len(categories) - 1:
            latex_table += "\\midrule\n"
    
    # End LaTeX table
    latex_table += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{small}\n"
        "\\end{center}\n"
        "\\end{table}\n"
    )
    
    return latex_table