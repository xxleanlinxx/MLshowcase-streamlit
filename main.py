# Basic
import streamlit as st
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML, XAI
import shap
from pdpbox import pdp
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (r2_score, classification_report, f1_score,
                             mean_absolute_error, recall_score, roc_auc_score)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor

# UI
from streamlit_option_menu import option_menu
#------------------------------------------------------------------------------------------------------#
st.set_page_config(layout = "wide")

st.header(" 🎛️ Demo of *Machine Learning Model* & *Explainable AI* ")
st.caption('''
*this app tries to standerdize a process of understanding a **Machine Learning Model** Performance with meaningful metrics & visualizations*
''')
st.logo("assets/button.png")

sns.set_theme(style = "whitegrid")
#------------------------------------------------------------------------------------------------------#

# Predefined dataset selection
dataset_options = ['mpg', 'titanic', 'secom']

# Dataset summaries
dataset_summaries = {
    'mpg': "Dataset about fuel efficiency of cars, with attributes such as miles per gallon (MPG), number of cylinders, horsepower, weight, model year, and origin. Often used for regression and exploratory data analysis.",
    'titanic': "Famous dataset on Titanic passengers, including attributes such as age, sex, class, survival status, and ticket price. Widely used for machine learning classification tasks and survival analysis.",
    'secom': "SECOM dataset from a semi-conductor manufacturing process. Contains 1567 samples with 590 sensor/process measurements. The task is binary classification (Pass/Fail) for in-house line testing yield, with a ~14:1 class imbalance (104 fails out of 1567). Classic use case for feature engineering, handling class imbalance, and ensemble learning."
}

# Dataset column descriptions
dataset_columns = {
    'mpg': {
        'mpg': "Miles per gallon, a measure of fuel efficiency.",
        'cylinders': "Number of cylinders in the car engine.",
        'displacement': "Engine displacement in cubic inches.",
        'horsepower': "Horsepower of the car.",
        'weight': "Weight of the car in pounds.",
        'acceleration': "Time to accelerate from 0 to 60 mph in seconds.",
        'model_year': "Year of the car's model (e.g., 70 for 1970).",
        'origin': "Origin of the car (1: USA, 2: Europe, 3: Japan).",
        'name': "Name of the car model."
    },
    'titanic': {
        'survived': "Survival status (0: No, 1: Yes).",
        'pclass': "Passenger class (1: First, 2: Second, 3: Third).",
        'sex': "Sex of the passenger (male or female).",
        'age': "Age of the passenger in years.",
        'sibsp': "Number of siblings/spouses aboard.",
        'parch': "Number of parents/children aboard.",
        'fare': "Fare amount paid in USD.",
        'embarked': "Port of embarkation (C: Cherbourg, Q: Queenstown, S: Southampton).",
        'class': "Passenger class as a string (First, Second, Third).",
        'who': "Categorical description of who (man, woman, child).",
        'deck': "Deck of the ship the passenger was on.",
        'embark_town': "Town where the passenger embarked.",
        'alive': "Survival status as a string (yes or no).",
        'alone': "Whether the passenger was alone (True or False)."
    },
    'secom': {
        'Sensor_0 ~ Sensor_N': "590 sensor/process measurement readings from the semiconductor manufacturing line. After feature engineering (removing high-missing, zero-variance, and highly correlated features), ~272 features remain.",
        'label': "Binary classification target (0: Pass, 1: Fail). -1 in raw data maps to Pass, 1 maps to Fail.",
        'status': "Pass/Fail status as a categorical string for EDA visualization."
    }
}

#------------------------------------------------------------------------------------------------------#


# Allow user to upload a file or choose a predefined dataset
with st.sidebar:
    selected_dataset = st.selectbox(
        "👾 *Choose a Dataset* ⤵️",
        ['-- null --'] + dataset_options  # Add 'None' for default empty selection
    )
    #------------------------------------------------------------------------------------------------------#

# Load the selected dataset or uploaded file
if selected_dataset != '-- null --':
    if selected_dataset == 'secom':
        with open("assets/secom_eda_data.pkl", "rb") as f:
            secom_eda_data = pickle.load(f)
        df = secom_eda_data['df']
        st.success(f"✅ Have Loaded <`{selected_dataset}`> dataset (pre-processed from raw SECOM data)!")
    else:
        df = sns.load_dataset(selected_dataset)
        st.success(f"✅ Have Loaded <`{selected_dataset}`> dataset from Seaborn!")
else:
    df = None
#------------------------------------------------------------------------------------------------------#

# Proceed only if a dataset is loaded
if df is not None:
    st.subheader("🕹️  *Switch Tab* ")

    # Option Menu
    with st.container():
        selected = option_menu(
            menu_title = None,
            options = ["Summary", "EDA Plot", "ML & XAI"],
            icons = ["blockquote-left", "bar-chart-line-fill", "diagram-3-fill"],
            orientation = 'horizontal'
        )
    
    if selected == "Summary":
        tab00, tab01, tab02, tab03 = st.tabs(['⌈⁰ Dataset Intro ⌉', 
                                              '⌈¹ Columns Info ⌉',
                                              '⌈² Dtypes Info ⌉', 
                                              '⌈³ Filter & View ⌉'])
        with tab00:
            st.subheader("🪄 Brief Intro to this Data")
            st.info(dataset_summaries[selected_dataset], icon = "ℹ️")
        #------------------------------------------------------------------------------------------------------#
        with tab01:
            if selected_dataset in dataset_columns:
                st.subheader("🪄 Definitions of the Columns")
                for col, desc in dataset_columns[selected_dataset].items():
                    st.markdown(f'''
                    **{col}**:
                    > *{desc}*
                    ''')
        #------------------------------------------------------------------------------------------------------#
        with tab02:
            st.warning(" Summary & Data types of the Dataset ", icon = "🕹️")
            st.info('Here is the Dataset', icon = "1️⃣")
            st.dataframe(df)
            
            st.divider()

            st.info('Data Type of Variables', icon = "2️⃣")
            
            # Data types overview
            data_types = df.dtypes.to_frame('Types')
            data_types['Types'] = data_types['Types'].astype(str)  # Convert to strings for sorting
            st.write(data_types.sort_values('Types'))
            
            st.divider()
        
            # Only describe numeric columns
            st.info('Statistic of `Numeric` Variables', icon = "3️⃣")
            numeric_df = df.select_dtypes(include = ['number'])
            if not numeric_df.empty:
                st.write(numeric_df.describe([.25, .75, .9, .95]))
            else:
                st.write("No numeric columns to describe.")
        #------------------------------------------------------------------------------------------------------#
        with tab03:
            st.warning(" Filter & View on Specific Column & Value ", icon = "🕹️")
            # Filter Data Section
            columns = df.columns.tolist()

            # Unique keys for selectbox
            selected_column = st.selectbox(
                'Select column to filter by',
                columns,
                key = 'column_selector_tab2',
            )
        
            if selected_column:
                # Show Filtered Data
                unique_values = df[selected_column].dropna().unique()  # Drop NaNs for filtering
                unique_values = [str(value) for value in unique_values]  # Ensure all values are string
                selected_value = st.selectbox(
                    'Select value',
                    unique_values,
                    key = 'value_selector_tab2',
                )

                st.divider()
                
                # Filter DataFrame
                st.info(f'Filtered Data of {selected_column} = {selected_value}', icon = "1️⃣")
                filtered_df = df[df[selected_column].astype(str) == selected_value]
                st.write("Filtered DataFrame:")
                st.write(filtered_df)
                
                st.divider()
                
                # Calculate Data Groupby Selected-Column
                st.info(f'Value Count Groupby {selected_column}', icon = "2️⃣")
                group_stats = df.groupby(selected_column).size().reset_index(name = 'counts')
                group_stats.set_index(selected_column, inplace = True)
                st.write(group_stats.sort_values('counts', ascending = False))
    #------------------------------------------------------------------------------------------------------#
    if selected == "EDA Plot":
        tab10, tab11, tab12, tab13, tab14 = st.tabs(['⌈⁰ ANOVA & 1 Categorical Plot ⌉', 
                                                     '⌈¹ Groupby 2+ Categorical Plot ⌉', 
                                                     '⌈² Cross 2 Numeric Plot ⌉', 
                                                     '⌈³ Diagnose Multi-Collinearity ⌉',
                                                     '⌈⁴ Overall Correlation ⌉'])
        #------------------------------------------------------------------------------------------------------#
        with tab10:
            st.markdown('''
                #### *One-way ANOVA & Violin Plot*
            ''')
            st.warning(" Testing the Statistically Significant Differences ", icon = "🕹️")
            
            # Filter numeric and categorical columns
            numeric_columns = df.select_dtypes(include = ['number']).columns.tolist()
            if selected_dataset == "mpg":
                categorical_columns = [col for col in df.select_dtypes(include=['object', 'category']).columns if col != 'name']
            else:
                categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()
            
            if numeric_columns and categorical_columns:
                # Allow user to select a categorical column and a numeric column
                selected_category_column = st.selectbox('Select `Categorical` Column',
                                                        categorical_columns,
                                                        key = 'category_selector_tab3',
                                                       )
                selected_numeric_column = st.selectbox('Select `Numeric` Column',
                                                       numeric_columns,
                                                       key = 'numeric_selector_tab3',
                                                      )

                st.divider()

                if selected_category_column and selected_numeric_column:
                    # #0 Check the Anova Test
                    # Remove rows with missing values in the selected columns
                    df = df.dropna(subset = [selected_numeric_column, selected_category_column])

                    # Ensure the data columns are of the correct type
                    df[selected_numeric_column] = pd.to_numeric(df[selected_numeric_column], errors = 'coerce')
                    df[selected_category_column] = df[selected_category_column].astype(str)

                    # Retrieve unique category values and group data by categories
                    unique_category_values = df[selected_category_column].unique().tolist()
                    category_groups = [df[df[selected_category_column] == category][selected_numeric_column] for category in unique_category_values]

                    # Check if each group has sufficient data
                    for i, group in enumerate(category_groups):
                        if len(group) < 2:
                            st.error(f"⛔ Group '{unique_category_values[i]}' does not have enough data for ANOVA analysis!")
                            st.stop()
                        if group.var() == 0:
                            st.error(f"⛔ Group '{unique_category_values[i]}' has constant values, making ANOVA analysis impossible!")
                            st.stop()

                    # Perform ANOVA
                    anova_result = f_oneway(*category_groups)

                    # Output the results
                    st.info(f'One-way ANOVA between {selected_category_column} on {selected_numeric_column}', icon = "ℹ️")
                    st.write(f"ANOVA F-statistic: {anova_result.statistic:.3f}")
                    st.write(f"ANOVA p-value: {anova_result.pvalue:.3f}")

                    if anova_result.pvalue < 0.05:
                        st.success("✅ The differences between groups `ARE` statistically significant (p < 0.05).")
                    else:
                        st.warning("⛔ The differences between groups are `NOT` statistically significant (p >= 0.05).")
                    
                    st.divider()
                    
                    # Violin plot
                    st.info(f'Violin plot of {selected_numeric_column} by {selected_category_column}', icon = "ℹ️")
                    fig, ax = plt.subplots(figsize = (12, 6))
                    sns.violinplot(
                        data = df,
                        x = selected_category_column,
                        y = selected_numeric_column,
                        palette = "muted",
                        ax = ax,
                    )
                    ax.set_xlabel(selected_category_column)
                    ax.set_ylabel(selected_numeric_column)
                    
                    st.pyplot(fig)

                    st.divider()
                    
                    # Calculate Statistics
                    st.info(f'Statistics of {selected_numeric_column} by {selected_category_column}', icon = "ℹ️")
                    grouped_stats = df.groupby(selected_category_column)[selected_numeric_column].agg(count = 'count',
                                                                                                      mean = 'mean',
                                                                                                      std = 'std',
                                                                                                      q1 = lambda x: x.quantile(0.25),
                                                                                                      median = 'median',
                                                                                                      q3 = lambda x: x.quantile(0.75),
                                                                                                      ).reset_index()

                    grouped_stats[['mean', 'std', 'q1', 'median', 'q3']] = grouped_stats[['mean', 'std', 'q1', 'median', 'q3']].round(3)
                
                    # Rename Columns of Statistics
                    grouped_stats.rename(columns = {'count': 'Count',
                                                    'mean': 'Mean',
                                                    'std': 'STD',
                                                    'q1': 'Q1','median': 'Q2',
                                                    'q3': 'Q3',
                                                    },
                                         inplace = True,
                                         )
                    grouped_stats.set_index(selected_category_column, inplace = True)
                    st.write(grouped_stats.T)

                    st.divider()
                    
                    df = df.dropna(subset = [selected_numeric_column, selected_category_column])
                    # Displot
                    st.info(f'Area Distribution of {selected_numeric_column} by {selected_category_column}', icon = "ℹ️")
                    sns_displot = sns.displot(data = df,
                                              x = selected_numeric_column,
                                              hue = selected_category_column,
                                              kind = "kde",
                                              height = 6,
                                              aspect = 1.5, # ratio of width:height = aspect
                                              multiple = "fill",
                                              clip = (0, None),
                                              palette = "ch:rot = -.25, hue = 1, light = .75",
                                              )

                    st.pyplot(sns_displot.fig)
            else:
                st.write("Ensure your dataset contains both numeric and categorical columns.", icon = "❗")
        #------------------------------------------------------------------------------------------------------#
        with tab11:
            st.markdown('''
                #### *Grouped split Violins & 3-way ANOVA*
            ''')
            st.warning(" Realize the Difference Accross Multiple Categorical Var ", icon = "🕹️")
            st.error(" If there's less than 2 `Categorical` Columns in the Dataset then this Tab is Unavailble ", icon = "⛔")
            
            # Filter numeric and categorical columns
            numeric_columns = df.select_dtypes(include = ['number']).columns.tolist()
            if selected_dataset == "mpg":
                categorical_columns = [col for col in df.select_dtypes(include=['object', 'category']).columns if col != 'name']
            else:
                categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()

            if numeric_columns and categorical_columns:
                # Allow user to select a categorical column and a numeric column
                selected_category_1 = st.selectbox('1️⃣ Select FIRST `Categorical` Column',
                                                   categorical_columns,
                                                   key = 'category_selector_1st_tab11',
                                                  )
                selected_category_2 = st.selectbox('2️⃣ Select SECOND `Categorical` Column',
                                                   categorical_columns,
                                                   key = 'category_selector_2nd_tab11',
                                                  )
                selected_category_3 = st.selectbox('3️⃣ Select `Categorical` Column for Further Testify',
                                                   categorical_columns,
                                                   key = 'category_selector_3rd_tab11',
                                                  )
                st.warning(" Selected `Categorical` Column Should be Different ", icon = "⚠️")
                selected_numeric_column = st.selectbox('♾️ Select `Numeric` Column',
                                                       numeric_columns,
                                                       key = 'numeric_selector_tab11',
                                                       )

                if selected_numeric_column and selected_category_1 and selected_category_2 and selected_category_3:
                    df = df.dropna(subset = [selected_numeric_column, selected_category_1, selected_category_2, selected_category_3])
                    # Split Violin
                    st.info(f'Split Violin of {selected_numeric_column} Groupby {selected_category_1} & {selected_category_2}', icon = "ℹ️")

                    fig, ax = plt.subplots(figsize = (12,6))
                    sns_splitviolin = sns.violinplot(data = df,
                                                     x = selected_category_1,
                                                     y = selected_numeric_column,
                                                     hue = selected_category_2,
                                                     split = True,
                                                     inner = "quart",
                                                     fill = False,
                                                     ax = ax,
                                                    )
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles[:len(set(df[selected_category_2]))], 
                              labels[:len(set(df[selected_category_2]))], 
                              title = selected_category_2,
                              loc = 'upper right', 
                              bbox_to_anchor = (1.2, 1))
                    
                    st.pyplot(fig)

                    st.divider()
                    
                    # 3-way ANOVA(Interaction plot)
                    st.info(' 3-way ANOVA Interaction Plot (only availalbe for `2+ Categorical` Var)', icon = "ℹ️")
                    sns_catplot = sns.catplot(data = df, 
                                              x = selected_category_1, 
                                              y = selected_numeric_column, 
                                              hue = selected_category_2, 
                                              col = selected_category_3,
                                              capsize = .2, palette = "YlGnBu_d", errorbar = "se",
                                              kind = "point", height = 6, aspect = .75,
                                             )
                    sns_catplot.despine(left = True)

                    st.pyplot(sns_catplot.fig)
            else:
                st.write("Ensure your dataset contains both `Numeric` and `Categorical` columns.", icon = "❗")
        #------------------------------------------------------------------------------------------------------#
        with tab12:
            st.markdown('''
                #### *2-Dimensional Density Plot*
            ''')
            st.warning(" Brief Realization on Correlation by Categorical Var Between Numeric Var ", icon = "🕹️")
            
            # Filter numeric columns
            numeric_columns = df.select_dtypes(include = ['number']).columns.tolist()

            # Filter categorical columns
            if selected_dataset == "mpg":
                categorical_columns = [col for col in df.select_dtypes(include=['object', 'category']).columns if col != 'name']
            else:
                categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()

            if numeric_columns and categorical_columns:
                # Allow user to select a categorical column
                selected_category_column = st.selectbox('Select `Categorical` Column',
                                                        categorical_columns,
                                                        key = 'category_selector_tab12',
                                                        )
                unique_category_values = df[selected_category_column].unique().tolist()

                # Allow user to select numeric columns for X and Y axes
                st.warning(" X & Y `Numeric` Should be Different ", icon = "⚠️")
                selected_x = st.selectbox('1️⃣ Select *X-axis* column `Numeric`',
                                          numeric_columns,
                                          key = 'x_axis_selector_tab12',
                                          )
                selected_y = st.selectbox('2️⃣ Select *Y-axis* column `Numeric`',
                                          numeric_columns,
                                          key = 'y_axis_selector_tab12',
                                          )
                if selected_x and selected_y:
                    # Create subplots based on the number of unique category values
                    num_categories = len(unique_category_values)
                    cols = 2  # Maximum 2 plots per row
                    rows = (num_categories + cols - 1) // cols  # Calculate rows needed

                    # Initialize the figure
                    fig, axes = plt.subplots(rows, cols,
                                             figsize = (12, 6 * rows),
                                             constrained_layout = True,
                                            )
                    axes = axes.flatten()  # Flatten axes for easy iteration

                    # Plot each category
                    for i, category in enumerate(unique_category_values):
                        ax = axes[i]
                        filtered_data = df[df[selected_category_column] == category]
                        sns.kdeplot(data = filtered_data,
                                    x = selected_x,
                                    y = selected_y,
                                    fill = True,
                                    cmap = "Greens",
                                    ax = ax,
                                    warn_singular = False,  # Suppress singular warnings
                                    )
                        ax.set_title(f'{selected_category_column}: {category}')
                        ax.set_xlabel(selected_x)
                        ax.set_ylabel(selected_y)

                    # Hide unused subplots
                    for i in range(num_categories, len(axes)):
                        axes[i].axis('off')
                    # Display the plot
                    st.pyplot(fig)
        #------------------------------------------------------------------------------------------------------#
        with tab13:
            st.markdown('''
                #### *Variance Inflation Factors(VIF) & Correlation Matrix Heatmap*
            ''')
            st.warning("Check the Multi-collinearity between Numeric Variables", icon = "🕹️")
            
            # Filter numeric columns
            numeric_columns = df.select_dtypes(include = ['number']).columns.tolist()
            
            if numeric_columns:
                # Put Numeric Var into Multi-Select
                default_cols = numeric_columns if len(numeric_columns) <= 15 else numeric_columns[:10]
                selected_columns = st.multiselect("Select `Numeric` columns:",
                                                  numeric_columns,
                                                  default = default_cols,
                                                  )
                st.divider()
                
                if selected_columns:
                    # VIF: Variance Inflation Factors
                    X = df[selected_columns].dropna()

                    # Add an Intercept
                    X = sm.add_constant(X)
                    
                    vif_data = pd.DataFrame()
                    vif_data["feature"] = X.columns
                    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    
                    st.info(' Use Variance Inflation Factors(`VIF`) to check `Multi-collinearity` ', icon = "ℹ️")
                    st.write(vif_data)
                    st.markdown('''
                                - VIF = 1: No multicollinearity.
                                - 1 < VIF < 5: Acceptable range.
                                - VIF ≥ 5 or 10: Severe multicollinearity; consider removing or combining features.
                    ''')
                    st.divider()

                    # Compute correlation matrix
                    correlation_matrix = df[selected_columns].corr()
        
                    # Mask to hide the upper triangle
                    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
                    # Plot the heatmap
                    fig, ax = plt.subplots(figsize = (12, 9))
                    sns.heatmap(correlation_matrix,
                                mask = mask,  # Apply the mask to hide the upper triangle
                                annot = True,
                                cmap = "coolwarm",
                                fmt = ".3f",
                                ax = ax,
                                )
                    ax.set_title("Correlation Matrix Heatmap (Lower Triangle Only)")
                    
                    st.info(' Use `Correlation Matrix Heatmap` for further checking ', icon = "ℹ️")
                    st.pyplot(fig)
                else:
                    st.warning("No columns selected. Please select at least one numeric column.", icon = "⚠️")
            else:
                st.error("Your dataset does not contain any numeric columns.", icon = "❗")
        #------------------------------------------------------------------------------------------------------#
        with tab14:
            st.markdown('''
                #### *Overall Pair plot*
            ''')
            st.warning(" Comparison between Numeric Var GroupBy Categorical Var  ", icon = "🕹️")
            st.success('''
                Pair plot is useful for:
                > - quickly exploring the relationships
                > - spotting correlations
                > - identifying any patterns or outliers
            ''')
            
            # Filter numeric and categorical columns
            numeric_columns = df.select_dtypes(include = ['number']).columns.tolist()
            if selected_dataset == "mpg":
                categorical_columns = [col for col in df.select_dtypes(include=['object', 'category']).columns if col != 'name']
            else:
                categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()

            if numeric_columns and categorical_columns:
                selected_category_column = st.selectbox(
                'Select Categorical Column',
                categorical_columns,
                key = 'category_selector_tab7',
                )

                if selected_category_column:
                    st.write(f"Selected Category: {selected_category_column}")

                    # Check if selected columns exist in df
                    if selected_category_column not in df.columns:
                        st.error(f"Column {selected_category_column} not found in dataframe.")
                    else:
                        # For high-dimensional datasets, let user pick columns
                        if len(numeric_columns) > 10:
                            st.warning("⚠️ Too many numeric columns for pair plot. Please select up to 10 columns.", icon = "⚠️")
                            plot_columns = st.multiselect(
                                "Select numeric columns for pair plot (max 10):",
                                numeric_columns,
                                default = numeric_columns[:5],
                                key = 'pairplot_columns_selector'
                            )
                            if len(plot_columns) > 10:
                                st.error("Please select at most 10 columns.")
                                plot_columns = plot_columns[:10]
                        else:
                            plot_columns = numeric_columns
                        
                        if plot_columns:
                            # Generate pairplot
                            pairplot_fig = sns.pairplot(df,
                                                        hue = selected_category_column,
                                                        vars = plot_columns,
                                                        corner = True,
                                                        plot_kws = {'alpha': 0.7},
                            )
                            
                            # Display the plot using Streamlit
                            st.pyplot(pairplot_fig)
            else:
                st.write("Ensure your dataset contains both numeric and categorical columns.", icon = "❗")
    #------------------------------------------------------------------------------------------------------#
    if selected == "ML & XAI":
        if selected_dataset == "secom":
            tab30, tab31, tab32, tab33 = st.tabs(['⌈⁰ Model Comparison ⌉',
                                                  '⌈¹ Feature Importance & Null Importance ⌉',
                                                  '⌈² Feature Engineering Pipeline ⌉',
                                                  '⌈³ SHAP Prediction on Sample ⌉'])
        else:
            tab30, tab31, tab32, tab33 = st.tabs(['⌈⁰ Model Summary ⌉',
                                                  '⌈¹ Feature Importance ⌉',
                                                  '⌈² Interaction Effect ⌉',
                                                  '⌈³ Prediction on Sample ⌉'])
        
        # ------------------------------------------- #
        # MPG (Regression)
        # ------------------------------------------- #
        if selected_dataset == "mpg":
            # ---------- (1) Import models & parameters ------------- #
            with open("assets/mpg_best_model.pkl", "rb") as f:
                best_model = pickle.load(f)
        
            with open("assets/mpg_explainer.pkl", "rb") as f:
                explainer = pickle.load(f)
        
            shap_values = np.load("assets/mpg_shap_values.npy", allow_pickle = True)
        
            with open("assets/mpg_best_params.pkl", "rb") as f:
                best_params = pickle.load(f)
        
            # ---------- (2) Loading Data & Pre-processing ---------- #
            df = sns.load_dataset('mpg')
            X = df.drop(columns = ["mpg", "name"])
            X = pd.get_dummies(X, drop_first = True)
            y = df["mpg"]
            
            # ---------- (3) Visualization ---------- #
            with tab30:
                st.caption("*Regression Showcase using The **Boosting** Method in Ensemble Learning*")
                st.write("### *LightGBM Regressor*")
                st.warning(" 🎖️ Prediction on the Fuel Efficiency of cars  `mpg`  (*Miles per Gallon*) ")
        
                y_pred = best_model.predict(X)
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                residuals = y - y_pred
        
                st.write("1️⃣", f"**R-squared**: *{r2:.2f}*")
                st.markdown(
                    """
                    - **R-squared** measures the proportion of variance in the target variable that is explained by the model.
                        > A score of 0.94 indicates that 94% of the variability in the target variable is explained by the model, which demonstrates a strong fit.
                    """)
                st.latex(r"R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}")

                st.divider()
                
                st.write("2️⃣", f"**Mean Residual**: *{np.mean(residuals):.2f}*")
                st.markdown(
                    """
                    - This represents the mean of the **Difference** between *Observed* and *Predicted* values.
                    - Use **Mean Residual** to check if the model has an overall bias on predict actual value.
                        - `≈0`: *No bias*, `>0`: *Underestimate*, `<0`: *Overestimate*
                        > A value close to 0 implies that the model's predictions, on average, are unbiased.
                    """)
                st.latex(r"\text{Mean Residual} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)")

                st.divider()
                
                st.write("3️⃣", f"**Mean Absolute Error (MAE)**: *{mae:.2f}*")
                st.markdown(
                    """
                    - MAE measures the **Average Magnitude of the Errors** in a set of predictions, without considering their direction
                        > An MAE of 1.37 indicates that the model's predictions deviate from the actual values by an average of 1.37 units.
                    """)
                st.latex(r"\text{MAE} = \frac{1}{n} \sum_{i=1}^n \left| y_i - \hat{y}_i \right|")

                st.divider()
                
                st.write("4️⃣", "**Best Model Parameters** *(GridSearchCV)*", best_params)
                st.markdown(
                    """
                    - *max_depth*
                        > This parameter controls the maximum depth of a tree. Limiting the depth helps reduce overfitting while maintaining model performance.
                    - *n_estimators* 
                        > This parameter specifies the number of boosting iterations (trees). A value of 100 strikes a balance between computational efficiency and prediction accuracy.
                    """)
        
            with tab31:
                st.caption("*Regression Showcase using The **Boosting** Method in Ensemble Learning*")
                st.write("### *Feature Importance Bar Chart*")
                st.info('''
                    ℹ️ Feature importance indicates *how much each feature contributes to the model's predictions* 
                    > Higher importance means the feature has a stronger influence on the outcome
                ''')
                
                feature_importances = np.mean(np.abs(shap_values), axis = 0)  # 計算特徵重要性 (平均絕對 SHAP 值)
                feature_names = X.columns
                
                sorted_idx = np.argsort(feature_importances)[::-1]
                sorted_importances = feature_importances[sorted_idx]
                sorted_features = feature_names[sorted_idx]
                
                colors = plt.cm.Greens(np.linspace(0.9, 0.2, len(sorted_importances)))
                
                fig_bar, ax_bar = plt.subplots(figsize = (10, 6))
                ax_bar.barh(
                    sorted_features,
                    sorted_importances,
                    color = colors,
                    edgecolor = 'black'
                )
                
                ax_bar.set_xlabel("Importance Score", fontsize = 12)
                ax_bar.set_ylabel("Features Name", fontsize = 12)
                ax_bar.invert_yaxis()
                
                st.pyplot(fig_bar)

                st.divider()
                
                st.write("### *SHAP Summary Plot*")

                st.info("ℹ️ This plot allows you to understand how the magnitude of a feature value influences the prediction ")
                st.success('''
                SHAP (**SH**apley **A**dditive ex**P**lanations) is a model explanation method based on *game theory*
                > It calculates the contribution of each feature to individual predictions and measures feature importance by averaging these contribution values.
                ''')
                
                fig_summary, ax_summary = plt.subplots(figsize = (10, 6))
                shap.summary_plot(shap_values, X, show = False)
                st.pyplot(fig_summary)

                st.divider()

                st.markdown('''
                #### Key Components of the SHAP Summary Plot
                
                ##### 1. **X-Axis (SHAP Values)**:
                - Represents the **magnitude and direction** of each feature's impact on the model's output.
                - **Positive SHAP values**: Feature contributes positively to the prediction (e.g., leaning towards a specific class).
                - **Negative SHAP values**: Feature contributes negatively to the prediction.
                
                ##### 2. **Y-Axis (Feature Names)**:
                - Displays the **features**, ranked by their importance.
                - *The most impactful features appear at the top.*
                
                ##### 3. **Point Distribution (Horizontal Spread)**:
                - Shows the **range of the feature's impact** across all samples.
                - A **wider spread** indicates the feature has more **variable impacts** on predictions.
                
                ##### 4. **Color (Feature Values)**:
                - The **color** of each point reflects the **actual feature value** for a given observation.
                - **Blue**: Low feature values.
                - **Red**: High feature values.
                ''')
        
            with tab32:
                st.caption("*Regression Showcase using The **Boosting** Method in Ensemble Learning*")
                
                feature_1 = st.selectbox("Select Feature 1:", X.columns)
                feature_2 = st.selectbox("Select Feature 2:", X.columns)

                st.divider()
        
                if feature_1 and feature_2:
                    st.write("### *Individual Conditional Expectation (ICE)*")
                    st.info('''
                        ℹ️ An ICE plot visualizes the effect of a single feature on the prediction for individual data points.
                        > While holding all the other feature **constant** values
                    ''')
                    st.success('''
                        - Each line represents how the model's prediction changes for a single data point as the chosen feature varies.
                        - Variation in line shapes indicates heterogeneity in the feature's effect.
                    ''')
                    
                    # Function to determine if a feature is binary
                    def is_binary(feature, data):
                        unique_values = data[feature].nunique()
                        return unique_values == 2
                
                    # Function to plot ICE or Partial Dependence based on feature type
                    def plot_feature(feature):
                        st.markdown(f"**Feature:** ***{feature}***")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        try:
                            if is_binary(feature, X):
                                st.warning(f"⚠️ **{feature}** is a binary feature. Displaying Average Partial Dependence Plot instead of ICE.")
                                PartialDependenceDisplay.from_estimator(
                                    estimator=best_model,
                                    X=X,
                                    features=[feature],
                                    kind="average",  # Use average partial dependence for binary features
                                    ax=ax,
                                    n_jobs=1  # Disable parallel processing for debugging
                                )
                            else:
                                PartialDependenceDisplay.from_estimator(
                                    estimator=best_model,
                                    X=X,
                                    features=[feature],
                                    kind="individual",  # ICE plot for non-binary features
                                    ax=ax,
                                    n_jobs=1  # Disable parallel processing for debugging
                                )
                            st.pyplot(fig)
                        except ValueError as ve:
                            st.error(f"ValueError while plotting for {feature}: {ve}")
                            st.text(traceback.format_exc())
                        except Exception as e:
                            st.error(f"An unexpected error occurred while plotting for {feature}: {e}")
                            st.text(traceback.format_exc())
                        finally:
                            plt.close(fig)  # Free up memory
                
                    # Plot both selected features
                    for feature in [feature_1, feature_2]:
                        plot_feature(feature)

                    st.divider()
                    
                    st.write("### *2-Dimensional Partial Dependence Plot (PDP)*")
                    st.info('''
                        ℹ️ 2D PDP plot shows how two features influence the predicted outcome of a machine learning model, while keeping all other features constant.
                        > This plot Helps identify *Interactions* between key features, providing valuable insights.
                    ''')
                    st.success('''
                        Color or Height represents the model's prediction value. 
                        - A *Smooth* surface suggests **minimal interaction** between the two features
                        - Distinct *Peaks* or *Valleys* indicate **significant interaction** effects
                    ''')
                    
                    fig_pdp, ax_pdp = plt.subplots(figsize = (10, 6))
                    PartialDependenceDisplay.from_estimator(
                        estimator = best_model,
                        X = X,
                        features = [(feature_1, feature_2)],
                        kind = "average",
                        ax = ax_pdp
                    )
                    st.pyplot(fig_pdp)
        
            with tab33:
                st.caption("*Regression Showcase using The **Boosting** Method in Ensemble Learning*")
                st.write("### *SHAP Waterfall Plot*")
                st.info("ℹ️ Waterfall plot illustrates how specific features contribute to the final prediction for a single instance in a machine learning model.")
        
                row_index = st.number_input("Select a Row Index of a Sample ⤵️", 
                                            min_value = 0, 
                                            max_value = len(X) - 1, 
                                            step = 1)
                if row_index is not None:
                    fig_waterfall, ax_waterfall = plt.subplots()
                    shap.waterfall_plot(
                        shap.Explanation(
                            base_values = explainer.expected_value,
                            values = shap_values[row_index],
                            data = X.iloc[row_index],
                            feature_names = X.columns.tolist()
                        ),
                        show = False
                    )
                    st.pyplot(fig_waterfall)
        
        # ------------------------------------------- #
        # Titanic (Classification)
        # ------------------------------------------- #
        elif selected_dataset == "titanic":
            # ---------- (1) Import models & parameters ------------- #
            with open("assets/titanic_best_model.pkl", "rb") as f:
                best_model = pickle.load(f)
        
            with open("assets/titanic_explainer.pkl", "rb") as f:
                explainer = pickle.load(f)
        
            shap_values = np.load("assets/titanic_shap_values.npy", allow_pickle = True)
        
            with open("assets/titanic_best_params.pkl", "rb") as f:
                best_params = pickle.load(f)
        
            # ---------- (2) Loading Data & Pre-processing ---------- #
            df = sns.load_dataset('titanic')
            
            valid_values = ["yes", "no"]
            df = df[df["alive"].isin(valid_values)]
            df = df.dropna(subset = ["alive"])
            df['alive'] = df['alive'].map({'yes': 1, 'no': 0})

            df['age'] = df['age'].fillna(df['age'].median())
            df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
            df['embark_town'] = df['embark_town'].fillna('Unknown')
            df['family_size'] = df['sibsp'] + df['parch'] + 1
            df = df.drop(columns = ['deck'])

            columns_to_drop = ['adult_male', 'who', 'survived', 'deck', 'embarked', 'pclass', 'alone', 'deck']
            df.drop(columns = [col for col in columns_to_drop if col in df.columns], inplace = True)
            df.dropna(axis = 0, how = "any")
            
            y = df["alive"]
            X = df.drop(columns = ["alive"])
            
            X = pd.get_dummies(X, drop_first = True)
        
            # ---------- (3) Visualization ---------- #
            with tab30:
                st.caption("*Classification Showcase using The **Bagging** Method in Ensemble Learning*")
                st.write("### *RandomForest Classifier*")
                st.warning(" 🎖️ Prediction on  `alived`  of Titanic passengers (*is alived or not*) ")
        
                y_pred = best_model.predict(X)
                report_dict = classification_report(y, y_pred, output_dict = True)
                cm = pd.DataFrame(report_dict).transpose()
                f1 = f1_score(y, y_pred)
        
                st.write("1️⃣", "**Confusion Matrix**:")
                st.write(cm)
                st.markdown("""
                    ##### *Confusion Matrix Metrics*
                    
                    The confusion matrix provides a detailed breakdown of the model's performance for each class:
                    
                    - *Precision*
                        > The proportion of correctly predicted positive observations to the total predicted positives:
                """)
                st.latex(r"\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}")
                
                st.markdown("""
                    - *Recall*
                        > The proportion of correctly predicted positive observations to all observations in the actual class:
                """)
                st.latex(r"\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}")
                
                st.markdown("""
                    - *F1-Score*
                        > The harmonic mean of precision and recall, balancing both metrics:
                """)
                st.latex(r"F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}")
                
                st.markdown("""
                    - *Support*
                       > The actual number of occurrences of each class in the dataset.
                    
                    ##### *Additional Metrics*
                    - *Accuracy*
                       > The overall percentage of correctly predicted observations.
                    - *Macro Average*
                       > Average performance across all classes, treating each class equally.
                    - *Weighted Average*
                       > Average performance weighted by the support of each class.
                """)

                st.divider()
                
                st.write("2️⃣", f"**F1-score**: *{f1:.3f}*")
                st.markdown(
                    """
                    The overall F1-score for the model is *0.916*, indicating strong balance between precision and recall.
                    """)

                st.divider()
                
                st.write("3️⃣", "**Best Model Parameters** *(GridSearchCV)*:", best_params)
                st.markdown(
                    """
                    - *max_depth*
                        > This parameter controls the maximum depth of a tree. Limiting the depth helps reduce overfitting while maintaining model performance.
                    - *n_estimators* 
                        > This parameter specifies the number of boosting iterations (trees). A value of 100 strikes a balance between computational efficiency and prediction accuracy.
                    """)
        
            with tab31:
                st.caption("*Classification Showcase using The **Bagging** Method in Ensemble Learning*")
                st.write("### *Feature Importance Bar Chart*")
                st.info('''
                    ℹ️ Feature importance indicates *how much each feature contributes to the model's predictions* 
                    > Higher importance means the feature has a stronger influence on the outcome
                ''')
                
                feature_importances = np.mean(np.abs(shap_values[:, :, 1]), axis = 0)  # 計算特徵重要性 (平均絕對 SHAP 值)
                feature_names = X.columns
                
                sorted_idx = np.argsort(feature_importances)[::-1]
                sorted_importances = feature_importances[sorted_idx]
                sorted_features = feature_names[sorted_idx]
                
                colors = plt.cm.Greens(np.linspace(0.9, 0.2, len(sorted_importances)))
                
                fig_bar, ax_bar = plt.subplots(figsize = (10, 6))
                ax_bar.barh(
                    sorted_features,
                    sorted_importances,
                    color = colors,
                    edgecolor = 'black'
                )
                
                ax_bar.set_xlabel("Importance Score", fontsize = 12)
                ax_bar.set_ylabel("Features Name", fontsize = 12)
                ax_bar.invert_yaxis()
                
                st.pyplot(fig_bar)
                
                st.divider()
                
                st.write("### *SHAP Summary Plot*")

                st.info("ℹ️ This plot allows you to understand how the magnitude of a feature value influences the prediction ")
                st.success('''
                SHAP (**SH**apley **A**dditive ex**P**lanations) is a model explanation method based on *game theory*
                > It calculates the contribution of each feature to individual predictions and measures feature importance by averaging these contribution values.
                ''')

                # st.success("✅ being a man (*who_man*) may negatively influence survival predictions (negative SHAP values), while being a woman (*who_woman*) has a positive influence.")
                # st.info("ℹ️ *age* plays an essential role in survival prediction, and higher ticket prices (*fare*) correlate with better survival odds.")

                fig_summary, ax_summary = plt.subplots(figsize = (10, 6))
                # with Two-classification: shap_values.shape = (n_samples, n_features, 2)
                shap.summary_plot(shap_values[:, :, 1], X, show = False)
                st.pyplot(fig_summary)

                st.divider()

                st.markdown('''
                #### Key Components of the SHAP Summary Plot
                
                ##### 1. **X-Axis (SHAP Values)**:
                - Represents the **magnitude and direction** of each feature's impact on the model's output.
                - **Positive SHAP values**: Feature contributes positively to the prediction (e.g., leaning towards a specific class).
                - **Negative SHAP values**: Feature contributes negatively to the prediction.
                
                ##### 2. **Y-Axis (Feature Names)**:
                - Displays the **features**, ranked by their importance.
                - *The most impactful features appear at the top.*
                
                ##### 3. **Point Distribution (Horizontal Spread)**:
                - Shows the **range of the feature's impact** across all samples.
                - A **wider spread** indicates the feature has more **variable impacts** on predictions.
                
                ##### 4. **Color (Feature Values)**:
                - The **color** of each point reflects the **actual feature value** for a given observation.
                - **Blue**: Low feature values.
                - **Red**: High feature values.
                ''')
        
            with tab32:
                st.caption("*Classification Showcase using The **Bagging** Method in Ensemble Learning*")

                feature_1 = st.selectbox("Select Feature 1:", X.columns)
                feature_2 = st.selectbox("Select Feature 2:", X.columns)
                
                st.divider()

                # PDP plot
                st.write("### *2-Dimensional Partial Dependence Plot (PDP)*")
                st.info('''
                    ℹ️ 2D PDP plot shows how two features influence the predicted outcome of a machine learning model, while keeping all other features constant.
                    > This plot Helps identify *Interactions* between key features, providing valuable insights.
                ''')
                st.success('''
                    Color or Height represents the model's prediction value. 
                    - A *Smooth* surface suggests **minimal interaction** between the two features
                    - Distinct *Peaks* or *Valleys* indicate **significant interaction** effects
                ''')
                
                fig_pdp, ax_pdp = plt.subplots(figsize = (10, 6))
                PartialDependenceDisplay.from_estimator(
                    estimator = best_model,
                    X = X,
                    features = [(feature_1, feature_2)],
                    kind = "average",
                    ax = ax_pdp
                )
                st.pyplot(fig_pdp)
        
            with tab33:
                st.caption("*Classification Showcase using The **Bagging** Method in Ensemble Learning*")
                st.write("### *SHAP Waterfall Plot*")
                st.info("ℹ️ Waterfall plot illustrates how specific features contribute to the final prediction for a single instance in a machine learning model.")
        
                row_index = st.number_input("Select a Row Index of a Sample ⤵️", 
                                            min_value = 0, 
                                            max_value = len(X) - 1, 
                                            step = 1)
                if row_index is not None:
                    fig_waterfall, ax_waterfall = plt.subplots()
                    shap.waterfall_plot(
                        shap.Explanation(
                            base_values = explainer.expected_value[1],
                            values = shap_values[1][row_index, :],
                            data = X.iloc[row_index, :],
                            feature_names = X.columns.tolist()
                        ),
                        show=False
                    )
                    st.pyplot(fig_waterfall)
        
        # ------------------------------------------- #
        # SECOM (Classification - XGBoost + LR Baseline)
        # ------------------------------------------- #
        elif selected_dataset == "secom":
            # ---------- (1) Import models & parameters ------------- #
            with open("assets/secom_model_results.pkl", "rb") as f:
                secom_results = pickle.load(f)

            with open("assets/secom_xgb_model.pkl", "rb") as f:
                best_model = pickle.load(f)

            with open("assets/secom_explainer.pkl", "rb") as f:
                explainer = pickle.load(f)

            shap_values = np.load("assets/secom_shap_values.npy", allow_pickle=True)

            with open("assets/secom_best_params.pkl", "rb") as f:
                best_params = pickle.load(f)

            # ---------- (2) Unpack results ---------- #
            X_test = secom_results['X_test']
            y_test = secom_results['y_test']
            feature_names = secom_results['feature_names']

            # ---------- (3) Tab 30: Model Comparison ---------- #
            with tab30:
                st.caption("*Classification Showcase: Logistic Regression (Baseline) vs XGBoost with SMOTE & Threshold Tuning*")
                st.write("### *Logistic Regression vs XGBoost Classifier*")
                st.warning(" 🎖️ Prediction on semiconductor manufacturing yield: `Pass` vs `Fail` (Binary Classification) ")

                st.divider()

                # Model Comparison Table
                st.info("1️⃣ **Model Performance Comparison** (Focus: Recall, F1, AUC-ROC)", icon="ℹ️")
                comparison_df = pd.DataFrame({
                    'Metric': ['Recall (Fail class)', 'F1-Score (Fail class)', 'AUC-ROC', 'Optimal Threshold'],
                    'Logistic Regression': [
                        f"{secom_results['lr_recall']:.3f}",
                        f"{secom_results['lr_f1']:.3f}",
                        f"{secom_results['lr_auc']:.3f}",
                        f"{secom_results['lr_optimal_threshold']:.3f}"
                    ],
                    'XGBoost': [
                        f"{secom_results['xgb_recall']:.3f}",
                        f"{secom_results['xgb_f1']:.3f}",
                        f"{secom_results['xgb_auc']:.3f}",
                        f"{secom_results['xgb_optimal_threshold']:.3f}"
                    ]
                }).set_index('Metric')
                st.dataframe(comparison_df, use_container_width=True)

                st.markdown("""
                > **Why not Accuracy?** With a ~14:1 class imbalance, a model predicting all samples as *Pass* 
                > would achieve ~93% accuracy but **0% Recall** on failures. In manufacturing, missing a defect 
                > (False Negative) is far more costly than a false alarm, so **Recall** is the primary metric.
                """)

                st.divider()

                # Classification Reports side by side
                st.info("2️⃣ **Detailed Classification Reports**", icon="ℹ️")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Logistic Regression (Baseline)**")
                    lr_report_df = pd.DataFrame(secom_results['lr_report']).transpose()
                    st.dataframe(lr_report_df.round(3))
                with col2:
                    st.write("**XGBoost**")
                    xgb_report_df = pd.DataFrame(secom_results['xgb_report']).transpose()
                    st.dataframe(xgb_report_df.round(3))

                st.divider()

                # Confusion Matrices
                st.info("3️⃣ **Confusion Matrices**", icon="ℹ️")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Logistic Regression**")
                    fig_cm_lr, ax_cm_lr = plt.subplots(figsize=(6, 4))
                    sns.heatmap(secom_results['lr_cm'], annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Pass', 'Fail'], yticklabels=['Pass', 'Fail'], ax=ax_cm_lr)
                    ax_cm_lr.set_xlabel('Predicted')
                    ax_cm_lr.set_ylabel('Actual')
                    st.pyplot(fig_cm_lr)
                with col2:
                    st.write("**XGBoost**")
                    fig_cm_xgb, ax_cm_xgb = plt.subplots(figsize=(6, 4))
                    sns.heatmap(secom_results['xgb_cm'], annot=True, fmt='d', cmap='Greens',
                                xticklabels=['Pass', 'Fail'], yticklabels=['Pass', 'Fail'], ax=ax_cm_xgb)
                    ax_cm_xgb.set_xlabel('Predicted')
                    ax_cm_xgb.set_ylabel('Actual')
                    st.pyplot(fig_cm_xgb)

                st.divider()

                # ROC Curves
                st.info("4️⃣ **ROC Curves Comparison**", icon="ℹ️")
                fig_roc, ax_roc = plt.subplots(figsize=(10, 6))
                ax_roc.plot(secom_results['lr_fpr'], secom_results['lr_tpr'],
                            label=f"Logistic Regression (AUC={secom_results['lr_auc']:.3f})", linestyle='--', color='steelblue')
                ax_roc.plot(secom_results['xgb_fpr'], secom_results['xgb_tpr'],
                            label=f"XGBoost (AUC={secom_results['xgb_auc']:.3f})", color='forestgreen', linewidth=2)
                ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('ROC Curve: LR vs XGBoost')
                ax_roc.legend(loc='lower right')
                ax_roc.grid(True, alpha=0.3)
                st.pyplot(fig_roc)

                st.divider()

                st.info("5️⃣ **Best XGBoost Parameters** *(GridSearchCV, scoring=recall)*", icon="ℹ️")
                st.write(best_params)
                st.markdown("""
                - *n_estimators*: Number of boosting rounds.
                - *max_depth*: Maximum tree depth — controls overfitting.
                - *learning_rate*: Step size shrinkage — smaller values require more boosting rounds.
                - *scale_pos_weight*: Balances the positive and negative class weights.
                """)

                st.divider()

                st.markdown("""
                ##### Key Formulas
                """)
                st.latex(r"\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}")
                st.latex(r"F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}")
                st.latex(r"\text{AUC-ROC} = \int_0^1 \text{TPR}(t)\, d(\text{FPR}(t))")

                st.divider()

                # Optimization Trend Chart
                if 'optimization_log' in secom_results:
                    st.info("6️⃣ **Multi-Round Optimization Journey**", icon="ℹ️")
                    st.markdown("""
                    > The following chart shows how **Recall** improved across iterative optimization rounds — 
                    > from a naive baseline to the final model achieving **≥95% Recall**.
                    """)

                    opt_log = secom_results['optimization_log']

                    # Trend line chart: Recall, F1, AUC across rounds
                    fig_trend, ax_trend = plt.subplots(figsize=(12, 6))
                    x_labels = opt_log['round_name'].tolist()
                    x_pos = np.arange(len(x_labels))

                    ax_trend.plot(x_pos, opt_log['recall'].values, 'o-', color='#e74c3c',
                                  linewidth=2.5, markersize=8, label='Recall', zorder=5)
                    ax_trend.plot(x_pos, opt_log['f1'].values, 's--', color='#3498db',
                                  linewidth=1.5, markersize=6, label='F1-Score', alpha=0.8)
                    ax_trend.plot(x_pos, opt_log['auc'].values, '^--', color='#2ecc71',
                                  linewidth=1.5, markersize=6, label='AUC-ROC', alpha=0.8)

                    # 95% target line
                    ax_trend.axhline(y=0.95, color='red', linestyle=':', linewidth=1.5,
                                     alpha=0.7, label='95% Recall Target')

                    # Annotate recall values
                    for i, (x, r) in enumerate(zip(x_pos, opt_log['recall'].values)):
                        ax_trend.annotate(f'{r:.3f}', (x, r), textcoords="offset points",
                                          xytext=(0, 12), ha='center', fontsize=9, fontweight='bold',
                                          color='#e74c3c')

                    ax_trend.set_xticks(x_pos)
                    ax_trend.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=9)
                    ax_trend.set_ylabel('Score', fontsize=12)
                    ax_trend.set_title('XGBoost Optimization Journey: Recall Improvement Across Rounds', fontsize=13)
                    ax_trend.legend(loc='lower right', fontsize=10)
                    ax_trend.set_ylim(0, 1.12)
                    ax_trend.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_trend)

                    st.divider()

                    # Optimization log table
                    st.write("**Detailed Optimization Log**")
                    display_log = opt_log[['round_name', 'description', 'recall', 'f1', 'auc',
                                           'precision', 'threshold']].copy()
                    display_log.columns = ['Round', 'Strategy', 'Recall', 'F1', 'AUC', 'Precision', 'Threshold']
                    st.dataframe(display_log.round(4), use_container_width=True, hide_index=True)

                    if 'winning_round' in secom_results:
                        st.success(f"🏆 **Final Winner: {secom_results['winning_round']}** — "
                                   f"Recall={secom_results['xgb_recall']:.3f}, "
                                   f"F1={secom_results['xgb_f1']:.3f}, "
                                   f"AUC={secom_results['xgb_auc']:.3f}, "
                                   f"Features={secom_results.get('final_feature_count', 'N/A')}")

                    # Generalization Diagnostics
                    if 'bootstrap_ci' in secom_results:
                        st.divider()
                        st.info("7️⃣ **Generalization Diagnostics** *(Bias-Variance & Honest Evaluation)*", icon="ℹ️")

                        boot = secom_results['bootstrap_ci']
                        cv_info = secom_results.get('cv_threshold_info', {})

                        col_boot1, col_boot2 = st.columns(2)
                        with col_boot1:
                            st.metric("Recall (Bootstrap Mean)", f"{boot['recall_mean']:.3f}",
                                      help="Mean recall across 1000 bootstrap resamples of the test set")
                            st.caption(f"95% CI: [{boot['recall_ci'][0]:.3f}, {boot['recall_ci'][1]:.3f}]")
                        with col_boot2:
                            st.metric("F1 (Bootstrap Mean)", f"{boot['f1_mean']:.3f}",
                                      help="Mean F1 across 1000 bootstrap resamples of the test set")
                            st.caption(f"95% CI: [{boot['f1_ci'][0]:.3f}, {boot['f1_ci'][1]:.3f}]")

                        st.markdown("""
                        **Why these diagnostics matter:**
                        - **Threshold** was selected via **5-fold CV on training data** — NOT tuned on test set (no data leakage)
                        - **Bootstrap CI** quantifies uncertainty due to the small test set (only 21 Fail samples)
                        - The CI lower bound shows the **worst-case recall** we can expect on unseen data
                        """)

                        if cv_info:
                            with st.expander("📋 CV Threshold Details (per model config)"):
                                for cname, cinfo in cv_info.items():
                                    st.write(f"**{cname}**")
                                    st.write(f"- CV median threshold: `{cinfo['cv_threshold']:.3f}`")
                                    st.write(f"- CV mean recall: `{cinfo['cv_mean_recall']:.3f}`")
                                    fold_str = ", ".join([f"{r:.3f}" for r in cinfo['fold_recalls']])
                                    st.write(f"- Fold recalls: [{fold_str}]")
                                    thr_str = ", ".join([f"{t:.3f}" for t in cinfo['fold_thresholds']])
                                    st.write(f"- Fold thresholds: [{thr_str}]")

            # ---------- (4) Tab 31: Feature Importance & Null Importance ---------- #
            with tab31:
                st.caption("*XGBoost Feature Importance via SHAP & Null Importance Assessment*")

                st.write("### *SHAP-based Feature Importance (Top 20)*")
                st.info("""
                    ℹ️ Feature importance indicates *how much each feature contributes to the model's predictions* 
                    > Higher importance means the feature has a stronger influence on the outcome
                """)

                shap_feat_imp = secom_results['shap_feature_importance']
                top_n = 20
                top_features = shap_feat_imp.head(top_n)

                colors = plt.cm.Greens(np.linspace(0.9, 0.2, len(top_features)))
                fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
                ax_bar.barh(
                    top_features.index[::-1],
                    top_features.values[::-1],
                    color=colors[::-1],
                    edgecolor='black'
                )
                ax_bar.set_xlabel("Mean |SHAP Value|", fontsize=12)
                ax_bar.set_ylabel("Feature", fontsize=12)
                ax_bar.set_title(f"Top {top_n} Features by SHAP Importance")
                st.pyplot(fig_bar)

                st.divider()

                st.write("### *SHAP Summary Plot*")
                st.info("ℹ️ This plot shows how the magnitude of a feature value influences the prediction")
                st.success("""
                SHAP (**SH**apley **A**dditive ex**P**lanations) is a model explanation method based on *game theory*
                > It calculates the contribution of each feature to individual predictions and measures feature importance by averaging these contribution values.
                """)

                fig_summary, ax_summary = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test, max_display=20, show=False)
                st.pyplot(fig_summary)

                st.divider()

                st.markdown("""
                #### Key Components of the SHAP Summary Plot
                
                ##### 1. **X-Axis (SHAP Values)**:
                - **Positive SHAP values**: Feature pushes prediction towards *Fail*.
                - **Negative SHAP values**: Feature pushes prediction towards *Pass*.
                
                ##### 2. **Y-Axis (Feature Names)**:
                - Features ranked by importance. *Most impactful at top.*
                
                ##### 3. **Color (Feature Values)**:
                - **Blue**: Low feature values. **Red**: High feature values.
                """)

                st.divider()

                # Null Importance Assessment
                st.write("### *Null Importance Assessment*")
                st.info("""
                    ℹ️ Null Importance tests whether a feature's importance is **statistically significant** 
                    by comparing it against importance scores from models trained on **shuffled (random) labels**.
                    > Features whose actual importance exceeds the 95th percentile of null importances are deemed *truly significant*.
                """)

                null_result = secom_results['null_importance_result']
                n_significant = int(null_result['is_significant'].sum())
                n_total = len(null_result)

                st.write(f"**Significant features**: {n_significant} / {n_total}")

                # Plot top 20 features: actual vs null 95th percentile
                top_null = null_result.head(top_n).copy()
                fig_null, ax_null = plt.subplots(figsize=(10, 8))
                x_pos = np.arange(len(top_null))
                width = 0.35
                ax_null.barh(x_pos + width/2, top_null['actual_importance'].values, width,
                             label='Actual Importance', color='forestgreen', edgecolor='black')
                ax_null.barh(x_pos - width/2, top_null['null_95th_percentile'].values, width,
                             label='Null 95th Percentile', color='lightcoral', edgecolor='black', alpha=0.7)
                ax_null.set_yticks(x_pos)
                ax_null.set_yticklabels(top_null['feature'].values)
                ax_null.invert_yaxis()
                ax_null.set_xlabel('Importance Score')
                ax_null.set_title(f'Top {top_n} Features: Actual vs Null Importance')
                ax_null.legend(loc='lower right')
                ax_null.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig_null)

                st.divider()

                # Show full null importance table
                st.write("**Full Null Importance Result Table** (sorted by actual importance)")
                st.dataframe(null_result.round(5), use_container_width=True, height=400)

            # ---------- (5) Tab 32: Feature Engineering Pipeline ---------- #
            with tab32:
                st.caption("*Feature Engineering Pipeline & Data Preprocessing Summary*")
                st.write("### *SECOM Feature Engineering Pipeline*")

                proc_info = secom_eda_data['processing_info']

                # Pipeline steps
                st.info("1️⃣ **Data Cleaning Summary**", icon="ℹ️")
                pipeline_df = pd.DataFrame({
                    'Step': [
                        'Original Features',
                        'After removing >50% missing',
                        'After removing zero-variance',
                        'After removing corr > 0.95',
                        'Final Features for Modeling'
                    ],
                    'Features': [
                        proc_info['original_shape'][1],
                        proc_info['original_shape'][1] - proc_info['n_high_missing_dropped'],
                        proc_info['original_shape'][1] - proc_info['n_high_missing_dropped'] - proc_info['n_zero_var_dropped'],
                        proc_info['cleaned_shape'][1],
                        proc_info['cleaned_shape'][1]
                    ],
                    'Dropped': [
                        '-',
                        f"{proc_info['n_high_missing_dropped']}",
                        f"{proc_info['n_zero_var_dropped']}",
                        f"{proc_info['n_high_corr_dropped']}",
                        '-'
                    ]
                })
                st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

                st.markdown("""
                **Feature Engineering Steps:**
                1. **Drop High Missing (>50%)**: Sensors with excessive missing readings provide unreliable information.
                2. **Drop Zero Variance**: Constant-value sensors carry no discriminative power.
                3. **Median Imputation**: Remaining missing values filled with feature median (robust to outliers).
                4. **Remove High Correlation (>0.95)**: Redundant sensors removed to reduce multicollinearity.
                """)

                st.divider()

                # Class Imbalance & SMOTE
                st.info("2️⃣ **Class Imbalance & SMOTE**", icon="ℹ️")
                st.write(f"**Original class ratio**: {proc_info['imbalance_ratio']} (Pass : Fail)")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Before SMOTE (Train Set)**")
                    before_smote = secom_results['train_before_smote']
                    fig_before, ax_before = plt.subplots(figsize=(6, 4))
                    bars = ax_before.bar(['Pass', 'Fail'],
                                         [before_smote['Pass'], before_smote['Fail']],
                                         color=['steelblue', 'coral'], edgecolor='black')
                    for bar in bars:
                        ax_before.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                                       f'{int(bar.get_height())}', ha='center', fontweight='bold')
                    ax_before.set_ylabel('Count')
                    ax_before.set_title('Before SMOTE')
                    st.pyplot(fig_before)

                with col2:
                    st.write("**After SMOTE (Train Set)**")
                    after_smote = secom_results['train_after_smote']
                    fig_after, ax_after = plt.subplots(figsize=(6, 4))
                    bars = ax_after.bar(['Pass', 'Fail'],
                                        [after_smote['Pass'], after_smote['Fail']],
                                        color=['steelblue', 'coral'], edgecolor='black')
                    for bar in bars:
                        ax_after.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                                      f'{int(bar.get_height())}', ha='center', fontweight='bold')
                    ax_after.set_ylabel('Count')
                    ax_after.set_title('After SMOTE')
                    st.pyplot(fig_after)

                st.markdown("""
                **SMOTE (Synthetic Minority Over-sampling Technique)**:
                > Generates synthetic samples for the minority class (Fail) by interpolating between existing 
                > minority samples and their nearest neighbors. Applied **only to the training set** to avoid data leakage.
                """)

                st.divider()

                # PCA Visualization
                st.info("3️⃣ **PCA Visualization**", icon="ℹ️")

                pca_2d_df = secom_eda_data['pca_2d_df']
                pca_ev = secom_eda_data['pca_explained_variance']
                pca_cv = secom_eda_data['pca_cumulative_variance']

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**PCA 2D Scatter (Pass vs Fail)**")
                    fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
                    for label, color, marker in [(0, 'steelblue', 'o'), (1, 'coral', 'x')]:
                        mask = pca_2d_df['label'] == label
                        name = 'Pass' if label == 0 else 'Fail'
                        ax_pca.scatter(pca_2d_df.loc[mask, 'PC1'], pca_2d_df.loc[mask, 'PC2'],
                                       c=color, label=name, alpha=0.6, s=30, marker=marker)
                    ax_pca.set_xlabel('PC1')
                    ax_pca.set_ylabel('PC2')
                    ax_pca.legend()
                    ax_pca.set_title('PCA 2D Projection')
                    ax_pca.grid(True, alpha=0.3)
                    st.pyplot(fig_pca)

                with col2:
                    st.write("**Cumulative Explained Variance**")
                    fig_ev, ax_ev = plt.subplots(figsize=(8, 6))
                    ax_ev.plot(range(1, len(pca_cv)+1), pca_cv, 'o-', color='forestgreen', markersize=4)
                    ax_ev.axhline(y=0.90, color='r', linestyle='--', alpha=0.7, label='90% threshold')
                    ax_ev.set_xlabel('Number of Components')
                    ax_ev.set_ylabel('Cumulative Explained Variance')
                    ax_ev.set_title('PCA Explained Variance')
                    ax_ev.legend()
                    ax_ev.grid(True, alpha=0.3)
                    st.pyplot(fig_ev)

                st.markdown("""
                **PCA (Principal Component Analysis)**:
                > Reduces the high-dimensional sensor data to a few principal components for visualization.
                > Note: PCA is used here for **EDA visualization only** — the model uses original (cleaned) features 
                > so that SHAP explanations remain interpretable at the individual sensor level.
                """)

                st.divider()

                # Missing Value Distribution (from original data)
                st.info("4️⃣ **Original Missing Value Distribution**", icon="ℹ️")
                missing_summary = secom_eda_data['missing_summary']
                missing_nonzero = missing_summary[missing_summary['missing_pct'] > 0].copy()

                fig_miss, ax_miss = plt.subplots(figsize=(12, 5))
                ax_miss.hist(missing_nonzero['missing_pct'] * 100, bins=30, color='steelblue',
                             edgecolor='black', alpha=0.8)
                ax_miss.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50% threshold (drop)')
                ax_miss.set_xlabel('Missing Percentage (%)')
                ax_miss.set_ylabel('Number of Features')
                ax_miss.set_title('Distribution of Missing Value Percentages (Original 590 Features)')
                ax_miss.legend()
                ax_miss.grid(True, alpha=0.3)
                st.pyplot(fig_miss)

                st.write(f"Features with any missing values: **{len(missing_nonzero)}** / {proc_info['original_shape'][1]}")

            # ---------- (6) Tab 33: SHAP Prediction on Sample ---------- #
            with tab33:
                st.caption("*XGBoost Classification with SHAP Explanations for Individual Samples*")
                st.write("### *SHAP Waterfall Plot*")
                st.info("ℹ️ Waterfall plot illustrates how specific sensor readings contribute to the final prediction for a single test sample.")

                row_index = st.number_input("Select a Row Index of a Test Sample ⤵️",
                                            min_value=0,
                                            max_value=len(X_test) - 1,
                                            step=1)

                if row_index is not None:
                    # Show sample prediction info
                    actual_label = int(y_test.iloc[row_index])
                    pred_prob = secom_results['xgb_y_prob'][row_index]
                    pred_label = int(secom_results['xgb_y_pred'][row_index])

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Actual", "Pass ✅" if actual_label == 0 else "Fail ❌")
                    col2.metric("Predicted", "Pass ✅" if pred_label == 0 else "Fail ❌")
                    col3.metric("Fail Probability", f"{pred_prob:.3f}")

                    st.divider()

                    fig_waterfall, ax_waterfall = plt.subplots()
                    shap.waterfall_plot(
                        shap.Explanation(
                            base_values=explainer.expected_value,
                            values=shap_values[row_index],
                            data=X_test.iloc[row_index],
                            feature_names=X_test.columns.tolist()
                        ),
                        max_display=15,
                        show=False
                    )
                    st.pyplot(fig_waterfall)

                    st.divider()

                    st.write("### *SHAP Force Plot*")
                    st.info("ℹ️ Force plot shows the same information in a horizontal layout — features pushing towards Fail (red) vs Pass (blue).")

                    fig_force = shap.force_plot(
                        explainer.expected_value,
                        shap_values[row_index],
                        X_test.iloc[row_index],
                        feature_names=X_test.columns.tolist(),
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(fig_force)

    #------------------------------------------------------------------------------------------------------#
else:
    st.error('''
    📎 Click TOP-LEFT **>** to GET STARTED
    ''')
    st.image('assets/diagram-export.png')
