import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import os

sns.set_theme(style='darkgrid')
sns.set_palette('viridis')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# Load and validate data
file_path = r'C:\Users\Owner\OneDrive\Documents\Air Pollution.csv'
try:
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        print("Please ensure 'Air Pollution.csv' is in 'C:\\Users\\Owner\\OneDrive\\Documents\\'")
        user_path = input("Enter the full path to 'Air Pollution.csv' (e.g., C:\\Users\\Owner\\Downloads\\Air Pollution.csv): ")
        if os.path.exists(user_path):
            df = pd.read_csv(user_path)
        else:
            raise FileNotFoundError(f"Could not find 'Air Pollution.csv' at {user_path}")
    else:
        df = pd.read_csv(file_path)

    # Print dataset info
    print('Dataset loaded successfully!')
    print('Columns:', df.columns.tolist())
    print('Sample data:\n', df.head())

    # Clean data
    df.replace('NA', np.nan, inplace=True)
    numeric_cols = ['pollutant_avg', 'pollutant_min', 'pollutant_max', 'latitude', 'longitude']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
except FileNotFoundError:
    print("Error: Could not load 'Air Pollution.csv'.")
    print("Please move it to 'C:\\Users\\Owner\\OneDrive\\Documents\\' or provide the correct path.")
    exit()
except Exception as e:
    print(f"Unexpected error loading data: {e}")
    exit()

# Check required columns
required_cols = ['state', 'pollutant_id', 'pollutant_avg']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Warning: Missing columns {missing_cols}. Some plots may fail.")

# Print data availability
print(f"Total rows: {len(df)}")
print(f"Non-NaN pollutant_avg: {df['pollutant_avg'].notna().sum()}")
print(f"Non-NaN pollutant_min: {df['pollutant_min'].notna().sum()}")
print(f"Non-NaN pollutant_max: {df['pollutant_max'].notna().sum()}")
print(f"Non-NaN latitude: {df['latitude'].notna().sum()}")
print(f"Pollutant IDs:\n{df['pollutant_id'].value_counts()}")
print(f"Top 5 states:\n{df['state'].value_counts().head(5)}")

# Plot functions with interactivity
def plot_histogram():
    try:
        if 'pollutant_avg' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            hist = sns.histplot(df['pollutant_avg'].dropna(), bins=30, kde=True, edgecolor='black', ax=ax)
            ax.set_title('Histogram', fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('Pollutant Average (µg/m³)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)

            def on_click(event):
                if event.inaxes == ax:
                    for patch in hist.patches:
                        if patch.contains(event)[0]:
                            height = patch.get_height()
                            x = patch.get_x() + patch.get_width() / 2
                            ax.annotate(f'Count: {int(height)}', (x, height), ha='center', va='bottom',
                                       fontsize=10, color='black', weight='bold')
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', on_click)
            plt.tight_layout()
            plt.savefig(r'C:\Users\Owner\OneDrive\Desktop\histogram.png', dpi=300)
            plt.show()
        else:
            print("Cannot plot Histogram: 'pollutant_avg' missing.")
    except Exception as e:
        print(f"Error in Histogram: {e}")

def plot_line():
    try:
        if 'state' in df.columns and 'pollutant_avg' in df.columns:
            state_data = df.groupby('state')['pollutant_avg'].mean().sort_values()
            fig, ax = plt.subplots(figsize=(12, 6))

            def animate(i):
                ax.clear()
                partial_data = state_data.iloc[:int(i)]
                sns.lineplot(x=partial_data.index, y=partial_data.values, marker='o', ax=ax)
                ax.set_title('Line Plot', fontsize=16, fontweight='bold', pad=15)
                ax.set_xlabel('State', fontsize=12)
                ax.set_ylabel('Mean Pollutant Average (µg/m³)', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

            ani = FuncAnimation(fig, animate, frames=len(state_data), interval=100, repeat=False)

            def on_motion(event):
                if event.inaxes == ax:
                    for i, (state, value) in enumerate(state_data.items()):
                        if abs(event.xdata - i) < 0.5:
                            ax.annotate(f'{state}: {value:.1f}', (i, value), ha='center', va='bottom',
                                       fontsize=10, color='black')
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect('motion_notify_event', on_motion)
            plt.tight_layout()
            plt.savefig(r'C:\Users\Owner\OneDrive\Desktop\line_plot.png', dpi=300)
            plt.show()
    except Exception as e:
        print(f"Error in Line Plot: {e}")

def plot_stacked_bar():
    try:
        if all(col in df.columns for col in ['state', 'pollutant_id', 'pollutant_avg']):
            pivot_table = df.pivot_table(values='pollutant_avg', index='state', columns='pollutant_id', aggfunc='mean')
            fig, ax = plt.subplots(figsize=(12, 6))
            pivot_table.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title('Stacked Bar Chart', fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('State', fontsize=12)
            ax.set_ylabel('Mean Pollutant Average (µg/m³)', fontsize=12)
            ax.legend(title='Pollutant', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)

            def on_click(event):
                if event.inaxes == ax:
                    for patch in ax.patches:
                        if patch.contains(event)[0]:
                            height = patch.get_height()
                            ax.annotate(f'{height:.1f}', (patch.get_x() + patch.get_width() / 2, patch.get_y() + height),
                                       ha='center', va='bottom', fontsize=10, color='black')
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', on_click)
            plt.tight_layout()
            plt.savefig(r'C:\Users\Owner\OneDrive\Desktop\stacked_bar_chart.png', dpi=300)
            plt.show()
        else:
            print("Cannot plot Stacked Bar Chart: Missing required columns.")
    except Exception as e:
        print(f"Error in Stacked Bar Chart: {e}")

def plot_heatmap():
    try:
        numeric_cols = [col for col in df.columns if col in ['latitude', 'longitude', 'pollutant_min', 'pollutant_max', 'pollutant_avg']]
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, ax=ax)
            ax.set_title('Heatmap', fontsize=16, fontweight='bold', pad=15)

            def on_motion(event):
                if event.inaxes == ax:
                    x, y = int(event.xdata), int(event.ydata)
                    if 0 <= x < len(corr_matrix) and 0 <= y < len(corr_matrix):
                        value = corr_matrix.iloc[y, x]
                        ax.annotate(f'{value:.2f}', (x + 0.5, y + 0.5), ha='center', va='center',
                                   fontsize=10, color='black', weight='bold')
                        fig.canvas.draw_idle()

            fig.canvas.mpl_connect('motion_notify_event', on_motion)
            plt.tight_layout()
            plt.savefig(r'C:\Users\Owner\OneDrive\Desktop\heatmap.png', dpi=300)
            plt.show()
        else:
            print("Cannot plot Heatmap: Not enough numeric columns.")
    except Exception as e:
        print(f"Error in Heatmap: {e}")

def plot_donut():
    try:
        if 'pollutant_id' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 8))
            pollutant_counts = df['pollutant_id'].value_counts()
            wedges, texts, autotexts = ax.pie(pollutant_counts, labels=pollutant_counts.index,
                                             autopct='%1.1f%%', wedgeprops={'width': 0.4})
            ax.set_title('Donut Chart', fontsize=16, fontweight='bold', pad=15)

            def on_click(event):
                if event.inaxes == ax:
                    for i, wedge in enumerate(wedges):
                        if wedge.contains_point((event.x, event.y)):
                            percentage = pollutant_counts.iloc[i] / pollutant_counts.sum() * 100
                            ax.annotate(f'{pollutant_counts.index[i]}: {percentage:.1f}%', 
                                       (0, 0), ha='center', fontsize=12, weight='bold')
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', on_click)
            plt.tight_layout()
            plt.savefig(r'C:\Users\Owner\OneDrive\Desktop\donut_chart.png', dpi=300)
            plt.show()
        else:
            print("Cannot plot Donut Chart: 'pollutant_id' missing.")
    except Exception as e:
        print(f"Error in Donut Chart: {e}")

def plot_grouped_bar():
    try:
        if all(col in df.columns for col in ['state', 'pollutant_id', 'pollutant_avg']):
            top_states = df['state'].value_counts().head(5).index
            pivot_grouped = df[df['state'].isin(top_states)].pivot_table(
                values='pollutant_avg', index='state', columns='pollutant_id', aggfunc='mean')
            fig, ax = plt.subplots(figsize=(14, 6))
            pivot_table = df.pivot_table(values='pollutant_avg', index='state', columns='pollutant_id', aggfunc='mean')
            pivot_table.plot(kind='bar', ax=ax)
            ax.set_title('Grouped Bar Chart', fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('State', fontsize=12)
            ax.set_ylabel('Mean Pollutant Average (µg/m³)', fontsize=12)
            ax.legend(title='Pollutant')
            plt.xticks(rotation=45)

            def on_click(event):
                if event.inaxes == ax:
                    for patch in ax.patches:
                        if patch.contains(event)[0]:
                            height = patch.get_height()
                            ax.annotate(f'{height:.1f}', (patch.get_x() + patch.get_width() / 2, patch.get_y() + height),
                                       ha='center', va='bottom', fontsize=10, color='black')
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', on_click)
            plt.tight_layout()
            plt.savefig(r'C:\Users\Owner\OneDrive\Desktop\grouped_bar_chart.png', dpi=300)
            plt.show()
        else:
            print("Cannot plot Grouped Bar Chart: Missing required columns.")
    except Exception as e:
        print(f"Error in Grouped Bar Chart: {e}")

def plot_box():
    try:
        if all(col in df.columns for col in ['pollutant_id', 'pollutant_avg']):
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x='pollutant_id', y='pollutant_avg', data=df, ax=ax)
            ax.set_title('Box Plot', fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('Pollutant Type', fontsize=12)
            ax.set_ylabel('Pollutant Average (µg/m³)', fontsize=12)
            plt.xticks(rotation=45)

            def on_motion(event):
                if event.inaxes == ax:
                    for i, artist in enumerate(ax.artists):
                        if artist.contains(event)[0]:
                            med = df[df['pollutant_id'] == df['pollutant_id'].unique()[i]]['pollutant_avg'].median()
                            ax.annotate(f'Median: {med:.1f}', (i, med), ha='center', va='bottom',
                                       fontsize=10, color='black')
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect('motion_notify_event', on_motion)
            plt.tight_layout()
            plt.savefig(r'C:\Users\Owner\OneDrive\Desktop\box_plot.png', dpi=300)
            plt.show()
        else:
            print("Cannot plot Box Plot: Missing 'pollutant_id' or 'pollutant_avg'.")
    except Exception as e:
        print(f"Error in Box Plot: {e}")

def plot_dashboard():
    try:
        if all(col in df.columns for col in ['pollutant_id', 'pollutant_avg', 'state', 'latitude']):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            sns.boxplot(x='pollutant_id', y='pollutant_avg', data=df, ax=ax1)
            ax1.set_title('Box Plot', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Pollutant', fontsize=10)
            ax1.set_ylabel('Avg (µg/m³)', fontsize=10)
            ax1.tick_params(axis='x', rotation=45)

            state_counts = df['state'].value_counts().head(10)
            ax2.bar(state_counts.index, state_counts.values)
            ax2.set_title('Bar Chart', fontsize=14, fontweight='bold')
            ax2.set_xlabel('State', fontsize=10)
            ax2.set_ylabel('Count', fontsize=10)
            ax2.tick_params(axis='x', rotation=45)

            sns.scatterplot(x='latitude', y='pollutant_avg', hue='pollutant_id', ax=ax3, data=df)
            ax3.set_title('Scatter Plot', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Latitude', fontsize=10)
            ax3.set_ylabel('Avg (µg/m³)', fontsize=10)
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            fig.suptitle('Dashboard', fontsize=16, fontweight='bold', y=1.05)

            def on_click(event):
                if event.inaxes:
                    ax = event.inaxes
                    ax.annotate(f'({event.xdata:.2f}, {event.ydata:.2f})', (event.xdata, event.ydata),
                               ha='center', va='bottom', fontsize=10, color='black')
                    fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', on_click)
            plt.tight_layout()
            plt.savefig(r'C:\Users\Owner\OneDrive\Desktop\dashboard.png', dpi=300)
            plt.show()
    except Exception as e:
        print(f"Error in Dashboard: {e}")

def plot_pair():
    try:
        numeric_cols = [col for col in df.columns if col in ['latitude', 'longitude', 'pollutant_min', 'pollutant_max', 'pollutant_avg']]
        if len(numeric_cols) >= 2:
            g = sns.pairplot(df[numeric_cols].dropna(), diag_kind='kde', plot_kws={'alpha': 0.6, 's': 50})
            g.fig.suptitle('Pair Plot', fontsize=16, fontweight='bold', y=1.02)

            def on_click(event):
                if event.inaxes:
                    ax = event.inaxes
                    x, y = event.xdata, event.ydata
                    ax.annotate(f'({x:.1f}, {y:.1f})', (x, y), ha='center', va='bottom',
                               fontsize=10, color='black')
                    g.fig.canvas.draw_idle()

            g.fig.canvas.mpl_connect('button_press_event', on_click)
            plt.savefig(r'C:\Users\Owner\OneDrive\Desktop\pair_plot.png', dpi=300)
            plt.show()
    except Exception as e:
        print(f"Error in Pair Plot: {e}")

# Console-based menu
def main_menu():
    plot_functions = {
        '1': (plot_histogram, "Histogram"),
        '2': (plot_line, "Line Plot"),
        '3': (plot_stacked_bar, "Stacked Bar Chart"),
        '4': (plot_heatmap, "Heatmap"),
        '5': (plot_donut, "Donut Chart"),
        '6': (plot_grouped_bar, "Grouped Bar Chart"),
        '7': (plot_box, "Box Plot"),
        '8': (plot_dashboard, "Dashboard"),
        '9': (plot_pair, "Pair Plot")
    }

    while True:
        print("\nAir Pollution Visualization Menu")
        for key, (_, name) in plot_functions.items():
            print(f"{key}. {name}")
        print("0. Exit")
        try:
            choice = input("Enter your choice (0-9): ").strip()
            if choice == '0':
                print("Exiting...")
                break
            elif choice in plot_functions:
                print(f"Generating {plot_functions[choice][1]}...")
                plot_functions[choice][0]()
            else:
                print("Invalid choice. Please enter 0-9.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error processing choice: {e}")

if __name__ == "__main__":
    main_menu()
