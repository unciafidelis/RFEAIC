import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
archivo_csv = 'data.csv'  # Make sure the CSV file is in the same directory or provide the full path
df = pd.read_csv(archivo_csv)

# Step 2: Verify the first few rows of the DataFrame
print(df.head())

# Step 3: Transform the data for the plot
# Create a DataFrame containing both original and synthetic values
df_orig = df.copy()
df_syn = df.copy()

# Modify the columns to include labels for Orig and Syn
df_orig['Type'] = 'Orig'
df_syn['Type'] = 'Syn'

# Combine Versus_CS and Cognitive_State into a single column for Y axis
df_orig['Y_Label'] = df_orig['Versus_CS'] + ' (' + df_orig['Cognitive_State'] + ')'
df_syn['Y_Label'] = df_syn['Versus_CS'] + ' (' + df_syn['Cognitive_State'] + ')'

# Concatenate the DataFrames to create a combined DataFrame
df_melted = pd.concat([
    df_orig[['Y_Label', 'Orig_Value', 'Type']].rename(columns={'Orig_Value': 'Value'}),
    df_syn[['Y_Label', 'Syn_Value', 'Type']].rename(columns={'Syn_Value': 'Value'})
])

# Step 4: Create a horizontal bar plot comparing Orig and Syn for each Y_Label
sns.set(style="whitegrid")
plt.figure(figsize=(16.5, 10))  # Increased figure size

# Create the horizontal bar plot
barplot = sns.barplot(data=df_melted, x='Value', y='Y_Label', hue='Type', errorbar=None, dodge=True)

# Añadir etiquetas a las barras
for p in barplot.patches:
    width = int(p.get_width())
    label = p.get_height()
    barplot.text(width + 1, p.get_y() + p.get_height() / 2, f'{width}', ha='center', va='center', color='black', size="8")
   
# Customize the plot
plt.title('Original and Syntethic Comparison by Cognitive State')
plt.xlabel('Number of subjets')
plt.legend(title='Type')

# Adjust layout to make sure labels are not cut off
plt.tight_layout()

# Show the plot
plt.show()
