import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Feedback Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS ===
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === DATA LOADING FUNCTION ===
@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv("feedback_parallel_output.csv")
        
        # Data preprocessing
        df['Severity'] = pd.to_numeric(df['Severity'], errors='coerce')
        df['Priority_numeric'] = df['Priority'].map({'High': 1, 'Medium': 2, 'Low': 3})
        df['Actionable_bool'] = df['Actionable'].map({'Yes': True, 'No': False})
        
        # Create severity labels
        severity_labels = {
            1: "Very Positive",
            2: "Mild Positive", 
            3: "Neutral",
            4: "Mild Negative",
            5: "Very Negative"
        }
        df['Severity_Label'] = df['Severity'].map(severity_labels)
        
        # Create impact score
        df['Impact_Score'] = df['Severity'] * (4 - df['Priority_numeric'])  # Higher severity + higher priority = higher impact
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# === MAIN APP ===
def main():
    st.markdown('<h1 class="main-header">📊 Employee Feedback Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # === SIDEBAR FILTERS ===
    with st.sidebar:
        st.markdown("## 🎛️ Filter Options")
        
        # Department filter
        departments = ['All'] + sorted(df['Department'].unique().tolist())
        selected_departments = st.multiselect("Select Department(s)", departments, default=['All'])
        
        # Benefit Type filter
        benefit_types = ['All'] + sorted(df['BenefitType'].unique().tolist())
        selected_benefit_types = st.multiselect("Select Benefit Type(s)", benefit_types, default=['All'])
        
        # Category filter
        categories = ['All'] + sorted(df['Category'].unique().tolist())
        selected_categories = st.multiselect("Select Category", categories, default=['All'])
        
        # Sentiment filter
        sentiments = ['All'] + sorted(df['Sentiment'].unique().tolist())
        selected_sentiments = st.multiselect("Select Sentiment", sentiments, default=['All'])
        
        # Actionable filter
        actionable_filter = st.radio("Show Actionable Items", ['All', 'Yes Only', 'No Only'])
        
        # Priority filter
        priorities = ['All'] + sorted(df['Priority'].unique().tolist())
        selected_priorities = st.multiselect("Select Priority", priorities, default=['All'])
        
        # Severity range
        severity_range = st.slider("Severity Range", 
                                 min_value=int(df['Severity'].min()), 
                                 max_value=int(df['Severity'].max()), 
                                 value=(int(df['Severity'].min()), int(df['Severity'].max())))
    
    # === APPLY FILTERS ===
    filtered_df = df.copy()
    
    if 'All' not in selected_departments:
        filtered_df = filtered_df[filtered_df['Department'].isin(selected_departments)]
    if 'All' not in selected_benefit_types:
        filtered_df = filtered_df[filtered_df['BenefitType'].isin(selected_benefit_types)]
    if 'All' not in selected_categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]
    if 'All' not in selected_sentiments:
        filtered_df = filtered_df[filtered_df['Sentiment'].isin(selected_sentiments)]
    if 'All' not in selected_priorities:
        filtered_df = filtered_df[filtered_df['Priority'].isin(selected_priorities)]
    
    if actionable_filter == 'Yes Only':
        filtered_df = filtered_df[filtered_df['Actionable_bool'] == True]
    elif actionable_filter == 'No Only':
        filtered_df = filtered_df[filtered_df['Actionable_bool'] == False]
    
    filtered_df = filtered_df[
        (filtered_df['Severity'] >= severity_range[0]) & 
        (filtered_df['Severity'] <= severity_range[1])
    ]
    
    # === KEY METRICS ===
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", len(filtered_df), delta=len(filtered_df)-len(df))
    
    with col2:
        actionable_count = len(filtered_df[filtered_df['Actionable_bool'] == True])
        st.metric("Actionable Items", actionable_count, 
                 delta=f"{actionable_count/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%")
    
    with col3:
        avg_severity = filtered_df['Severity'].mean() if len(filtered_df) > 0 else 0
        st.metric("Avg Severity", f"{avg_severity:.2f}", 
                 delta=f"{avg_severity-df['Severity'].mean():.2f}")
    
    with col4:
        negative_sentiment = len(filtered_df[filtered_df['Sentiment'] == 'Negative'])
        st.metric("Negative Feedback", negative_sentiment,
                 delta=f"{negative_sentiment/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%")
    
    with col5:
        high_priority = len(filtered_df[filtered_df['Priority'] == 'High'])
        st.metric("High Priority", high_priority,
                 delta=f"{high_priority/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%")
    
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters. Please adjust your selection.")
        return
    
    # === MAIN DASHBOARD TABS ===
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", "🔍 Detailed Analysis", "🎯 Priority Matrix", 
        "💡 Insights & Recommendations", "📋 Data Explorer", "📊 Advanced Analytics"
    ])
    
    with tab1:
        overview_tab(filtered_df)
    
    with tab2:
        detailed_analysis_tab(filtered_df)
    
    with tab3:
        priority_matrix_tab(filtered_df)
    
    with tab4:
        insights_recommendations_tab(filtered_df)
    
    with tab5:
        data_explorer_tab(filtered_df)
    
    with tab6:
        advanced_analytics_tab(filtered_df)

def overview_tab(df):
    """Overview dashboard with key visualizations"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment Distribution
        st.subheader("💭 Sentiment Distribution")
        sentiment_counts = df['Sentiment'].value_counts()
        
        fig_sentiment = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color_discrete_map={'Positive': '#00CC96', 'Neutral': '#FFA15A', 'Negative': '#EF553B'},
            title="Overall Sentiment Breakdown"
        )
        fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # Category Distribution
        st.subheader("📂 Category Distribution")
        category_counts = df['Category'].value_counts()
        
        fig_category = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            color=category_counts.values,
            color_continuous_scale='viridis',
            title="Feedback by Category"
        )
        fig_category.update_layout(showlegend=False)
        st.plotly_chart(fig_category, use_container_width=True)
    
    # Severity Analysis
    st.subheader("⚠️ Severity Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Severity by Department
        severity_dept = df.groupby(['Department', 'Severity_Label']).size().reset_index(name='Count')
        
        fig_severity_dept = px.bar(
            severity_dept,
            x='Department',
            y='Count',
            color='Severity_Label',
            color_discrete_map={
                'Very Positive': '#00CC96',
                'Mild Positive': '#7CFC00',
                'Neutral': '#FFA500',
                'Mild Negative': '#FF6347',
                'Very Negative': '#FF0000'
            },
            title="Severity Distribution by Department"
        )
        fig_severity_dept.update_xaxes(tickangle=45)
        st.plotly_chart(fig_severity_dept, use_container_width=True)
    
    with col4:
        # Priority vs Actionable
        priority_actionable = pd.crosstab(df['Priority'], df['Actionable'])
        
        # Reshape data for proper legend labels
        priority_data = pd.DataFrame({
            'Priority': priority_actionable.index.tolist() * 2,
            'Count': priority_actionable['Yes'].tolist() + priority_actionable['No'].tolist(),
            'Actionable': ['Actionable'] * len(priority_actionable.index) + ['Not Actionable'] * len(priority_actionable.index)
        })
        
        fig_priority = px.bar(
            priority_data,
            x='Priority',
            y='Count',
            color='Actionable',
            title="Priority vs Actionable Items",
            labels={'Priority': 'Priority Level', 'Count': 'Number of Items'},
            color_discrete_map={'Actionable': '#1f77b4', 'Not Actionable': '#ff7f0e'},
            barmode='stack'
        )
        fig_priority.update_layout(
            legend=dict(
                title="Item Status",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        st.plotly_chart(fig_priority, use_container_width=True)

def detailed_analysis_tab(df):
    """Detailed analysis with multiple visualization types"""
    
    # Benefit Type Analysis
    st.subheader("🎁 Benefit Type Analysis")
    
    # Sunburst chart for BenefitType -> BenefitSubType -> Severity
    benefit_analysis = df.groupby(['BenefitType', 'BenefitSubType', 'Severity_Label']).size().reset_index(name='Count')
    
    fig_sunburst = px.sunburst(
        benefit_analysis,
        path=['BenefitType', 'BenefitSubType', 'Severity_Label'],
        values='Count',
        color='Count',
        color_continuous_scale='RdYlBu_r',
        title='Benefit Analysis: Type → SubType → Severity'
    )
    fig_sunburst.update_traces(textinfo="label+percent entry")
    st.plotly_chart(fig_sunburst, use_container_width=True)
    
    # Detailed breakdown by subtype
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 SubType Performance")
        subtype_metrics = df.groupby('BenefitSubType').agg({
            'Severity': 'mean',
            'Impact_Score': 'mean',
            'Actionable_bool': 'sum'
        }).round(2).sort_values('Severity', ascending=False)
        
        subtype_metrics.columns = ['Avg Severity', 'Avg Impact', 'Actionable Count']
        st.dataframe(subtype_metrics, use_container_width=True)
    
    with col2:
        st.subheader("🔥 Top Issues by SubType")
        top_issues = df[df['Severity'] >= 4].groupby('BenefitSubType').size().sort_values(ascending=False).head(10)
        
        fig_issues = px.bar(
            x=top_issues.values,
            y=top_issues.index,
            orientation='h',
            color=top_issues.values,
            color_continuous_scale='Reds',
            title="Most Problematic SubTypes"
        )
        st.plotly_chart(fig_issues, use_container_width=True)
    
    # # Time series analysis (if date columns exist)
    # # === Department Sentiment Distribution ===
    # st.subheader("🏢 Department-wise Sentiment Distribution")
    
    # # Define standard sentiment categories
    # sentiment_categories = ['Positive', 'Neutral', 'Negative']
    
    # # Group and ensure all sentiment categories are present
    # dept_sentiment = (
    #     df.groupby(['Department', 'Sentiment'])
    #     .size()
    #     .unstack(fill_value=0)
    #     .reindex(columns=sentiment_categories, fill_value=0)
    # )
    
    # # Calculate percentage for Positive and Negative
    # dept_sentiment['Total'] = dept_sentiment.sum(axis=1)
    # dept_sentiment['Positive_%'] = (dept_sentiment['Positive'] / dept_sentiment['Total'] * 100).round(2)
    # dept_sentiment['Negative_%'] = (dept_sentiment['Negative'] / dept_sentiment['Total'] * 100).round(2)
    
    # # Remove departments with no feedback
    # dept_sentiment = dept_sentiment[dept_sentiment['Total'] > 0]
    
    # # Melt for grouped bar chart
    # dept_melt = dept_sentiment.reset_index()[['Department', 'Positive_%', 'Negative_%']].melt(
    #     id_vars='Department',
    #     var_name='SentimentType',
    #     value_name='Percentage'
    # )
    
    # # Plot grouped bar chart
    # fig = px.bar(
    #     dept_melt,
    #     x='Percentage',
    #     y='Department',
    #     color='SentimentType',
    #     barmode='group',
    #     orientation='h',
    #     color_discrete_map={
    #         'Positive_%': 'green',
    #         'Negative_%': 'red'
    #     },
    #     title="📊 Department-wise Positive vs Negative Feedback Percentage"
    # )
    
    # fig.update_layout(
    #     xaxis_title="Percentage (%)",
    #     yaxis_title="Department",
    #     legend_title="Sentiment Type"
    # )
    
    # st.plotly_chart(fig, use_container_width=True)
    
    # === Summary Metrics ===
    # if len(dept_sentiment) > 0:
    #     worst_dept_row = dept_sentiment.sort_values('Negative_%', ascending=False).iloc[0]
    #     worst_dept = worst_dept_row.name
    #     worst_neg_rate = worst_dept_row['Negative_%']
    #     avg_pos_rate = dept_sentiment['Positive_%'].mean()
    #     avg_neg_rate = dept_sentiment['Negative_%'].mean()
    
    #     col1, col2, col3 = st.columns(3)
    #     with col1:
    #         st.metric("📉 Avg Negative Rate", f"{avg_neg_rate:.1f}%")
    #     with col2:
    #         st.metric("📈 Avg Positive Rate", f"{avg_pos_rate:.1f}%")
    #     with col3:
    #         st.metric("🚨 Worst Performing Dept", f"{worst_dept}", f"{worst_neg_rate:.1f}%")
    # else:
    #     st.warning("No department data available with current filters.")



def priority_matrix_tab(df):
    """Priority matrix and impact analysis"""
    
    st.subheader("🎯 Impact vs Feasibility Matrix")
    
    # Filter for actionable items only
    actionable_df = df[df['Actionable_bool'] == True].copy()
    
    if len(actionable_df) == 0:
        st.warning("No actionable items in the current filter selection.")
        return
    
    # Create feasibility score (inverse of priority - higher priority = lower feasibility number)
    actionable_df['Feasibility'] = actionable_df['Priority_numeric'].map({1: 3, 2: 2, 3: 1})  # High priority = hard to implement
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bubble chart for impact vs feasibility
        fig_matrix = px.scatter(
            actionable_df,
            x='Feasibility',
            y='Severity',
            size='Impact_Score',
            color='BenefitType',
            hover_data=['BenefitSubType', 'Department', 'Priority'],
            title='Impact vs Feasibility Matrix',
            labels={
                'Severity': 'Impact (Severity Level)',
                'Feasibility': 'Feasibility (1=Hard, 3=Easy)'
            }
        )
        
        # Add quadrant lines
        fig_matrix.add_hline(y=3.5, line_dash="dash", line_color="gray", annotation_text="High Impact Threshold")
        fig_matrix.add_vline(x=2, line_dash="dash", line_color="gray", annotation_text="Feasibility Threshold")
        
        st.plotly_chart(fig_matrix, use_container_width=True)
    
    with col2:
        # Priority ranking
        st.subheader("🏆 Priority Ranking")
        
        priority_ranking = actionable_df.nlargest(15, 'Impact_Score')[
            ['BenefitSubType', 'Department', 'Priority', 'Severity', 'Impact_Score', 'SuggestedAction']
        ].reset_index(drop=True)
        
        priority_ranking.index += 1
        st.dataframe(priority_ranking, use_container_width=True)
    
    # Action prioritization
    st.subheader("📋 Recommended Actions by Quadrant")
    
    # Define quadrants
    high_impact_easy = actionable_df[(actionable_df['Severity'] >= 4) & (actionable_df['Feasibility'] >= 2)]
    high_impact_hard = actionable_df[(actionable_df['Severity'] >= 4) & (actionable_df['Feasibility'] < 2)]
    low_impact_easy = actionable_df[(actionable_df['Severity'] < 4) & (actionable_df['Feasibility'] >= 2)]
    low_impact_hard = actionable_df[(actionable_df['Severity'] < 4) & (actionable_df['Feasibility'] < 2)]
    
    quad1, quad2, quad3, quad4 = st.columns(4)
    
    with quad1:
        st.markdown("### 🟢 Quick Wins")
        st.markdown("*High Impact, Easy Implementation*")
        if len(high_impact_easy) > 0:
            for idx, row in high_impact_easy.head(3).iterrows():
                st.markdown(f"• **{row['BenefitSubType']}** ({row['Department']})")
        else:
            st.markdown("No items in this quadrant")
    
    with quad2:
        st.markdown("### 🟡 Major Projects")
        st.markdown("*High Impact, Hard Implementation*")
        if len(high_impact_hard) > 0:
            for idx, row in high_impact_hard.head(3).iterrows():
                st.markdown(f"• **{row['BenefitSubType']}** ({row['Department']})")
        else:
            st.markdown("No items in this quadrant")
    
    with quad3:
        st.markdown("### 🔵 Fill-ins")
        st.markdown("*Low Impact, Easy Implementation*")
        if len(low_impact_easy) > 0:
            for idx, row in low_impact_easy.head(3).iterrows():
                st.markdown(f"• **{row['BenefitSubType']}** ({row['Department']})")
        else:
            st.markdown("No items in this quadrant")
    
    with quad4:
        st.markdown("### 🔴 Questionable")
        st.markdown("*Low Impact, Hard Implementation*")
        if len(low_impact_hard) > 0:
            for idx, row in low_impact_hard.head(3).iterrows():
                st.markdown(f"• **{row['BenefitSubType']}** ({row['Department']})")
        else:
            st.markdown("No items in this quadrant")

def insights_recommendations_tab(df):
    """Insights and recommendations based on data analysis"""
    
    st.subheader("💡 Key Insights & Strategic Recommendations")
    
    # Generate insights
    actionable_items = df[df['Actionable_bool'] == True]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Key Findings")
        
        # Top problematic areas
        problem_areas = df[df['Severity'] >= 4].groupby('BenefitSubType').size().sort_values(ascending=False).head(5)
        
        st.markdown("#### 🚨 Most Problematic Benefit Areas:")
        for subtype, count in problem_areas.items():
            percentage = (count / len(df)) * 100
            st.markdown(f"• **{subtype}**: {count} issues ({percentage:.1f}% of total feedback)")
        
        # Department performance
        st.markdown("#### 📊 Department Performance:")
        dept_negative = df[df['Sentiment'] == 'Negative'].groupby('Department').size().sort_values(ascending=False).head(3)
        for dept, count in dept_negative.items():
            total_dept = len(df[df['Department'] == dept])
            rate = (count / total_dept) * 100
            st.markdown(f"• **{dept}**: {rate:.1f}% negative feedback rate")
    
    with col2:
        st.markdown("### 🎯 Strategic Recommendations")
        
        # Cutting recommendations
        cut_candidates = df[
            (df['Severity'] >= 4) & 
            (df['Sentiment'] == 'Negative') & 
            (df['Priority'] == 'Medium')
        ].groupby('BenefitSubType').size().sort_values(ascending=False).head(3)
        
        if len(cut_candidates) > 0:
            st.markdown("#### ✂️ Consider Optimizing:")
            for subtype, count in cut_candidates.items():
                avg_severity = df[df['BenefitSubType'] == subtype]['Severity'].mean()
                st.markdown(f"• **{subtype}** (Avg severity: {avg_severity:.1f}, {count} complaints)")
        
        # Enhancement recommendations
        enhance_candidates = actionable_items[
            actionable_items['Severity'] >= 4
        ].groupby('BenefitSubType').size().sort_values(ascending=False).head(3)
        
        if len(enhance_candidates) > 0:
            st.markdown("#### 🔧 Priority Enhancements:")
            for subtype, count in enhance_candidates.items():
                st.markdown(f"• **{subtype}** ({count} actionable improvements)")
    
    # Detailed recommendations table
    st.subheader("📋 Detailed Action Plan")
    
    # Create comprehensive recommendation table
    recommendations_df = actionable_items.groupby('BenefitSubType').agg({
        'Severity': 'mean',
        'Priority': lambda x: 'High' if 'High' in x.values else 'Medium',
        'Impact_Score': 'mean',
        'SuggestedAction': lambda x: '; '.join(x.unique()[:3])  # Top 3 unique actions
    }).round(2).sort_values('Impact_Score', ascending=False)
    
    recommendations_df.columns = ['Avg Severity', 'Priority Level', 'Impact Score', 'Suggested Actions']
    
    # Add recommendation category
    def get_recommendation(row):
        if row['Avg Severity'] >= 4.5:
            return "🔴 Critical - Immediate Action"
        elif row['Avg Severity'] >= 4.0:
            return "🟡 High Priority - Plan for Q1"
        elif row['Avg Severity'] >= 3.5:
            return "🔵 Medium Priority - Plan for Q2"
        else:
            return "🟢 Low Priority - Monitor"
    
    recommendations_df['Recommendation'] = recommendations_df.apply(get_recommendation, axis=1)
    
    st.dataframe(recommendations_df, use_container_width=True)
    
    # Download recommendations
    csv = recommendations_df.to_csv(index=True)
    st.download_button(
        label="📥 Download Action Plan",
        data=csv,
        file_name="benefit_recommendations.csv",
        mime="text/csv"
    )

def data_explorer_tab(df):
    """Interactive data explorer"""
    
    st.subheader("🔍 Data Explorer")
    
    # Search functionality
    search_term = st.text_input("🔍 Search in comments:", placeholder="Enter keywords to search...")
    
    if search_term:
        search_df = df[df['Comments'].str.contains(search_term, case=False, na=False)]
        st.write(f"Found {len(search_df)} records containing '{search_term}'")
    else:
        search_df = df
    
    # Column selector
    available_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to display:",
        available_columns,
        default=['EmployeeID', 'BenefitType', 'BenefitSubType', 'Sentiment', 'Severity', 'Priority', 'Actionable']
    )
    
    if selected_columns:
        # Display data
        st.dataframe(
            search_df[selected_columns],
            use_container_width=True,
            height=400
        )
        
        # Download filtered data
        csv = search_df[selected_columns].to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_feedback_data.csv",
            mime="text/csv"
        )
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Numerical Columns")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            st.dataframe(df[numerical_cols].describe(), use_container_width=True)
    
    with col2:
        st.markdown("#### Other Columns")
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_summary = {}
            for col in categorical_cols[:]:  # Show first 5 categorical columns
                cat_summary[col] = {
                    'Unique Values': df[col].nunique(),
                    'Most Common': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
                }
            st.dataframe(pd.DataFrame(cat_summary).T, use_container_width=True)

def advanced_analytics_tab(df):
    """Advanced analytics and correlations"""
    
    st.subheader("🧮 Advanced Analytics")
    
    # Correlation analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔗 Correlation Matrix")
        
        # Prepare numerical data for correlation
        corr_df = df[['Severity', 'Priority_numeric', 'Impact_Score']].corr()
        
        fig_corr = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Distribution Analysis")
        
        selected_metric = st.selectbox("Select metric for distribution:", 
                                     ['Severity', 'Impact_Score'])
        
        fig_dist = px.histogram(
            df,
            x=selected_metric,
            nbins=20,
            color='Sentiment',
            title=f"{selected_metric} Distribution by Sentiment"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Advanced insights
    st.subheader("🎯 Advanced Insights")
    
    # Sentiment vs BenefitType analysis
    sentiment_benefit_matrix = pd.crosstab(df['BenefitType'], df['Sentiment'], normalize='index') * 100
    
    fig_heatmap = px.imshow(
        sentiment_benefit_matrix,
        text_auto='.1f',
        aspect="auto",
        color_continuous_scale='RdYlGn',
        title="BenefitType Sentiment Distribution (%)",
        labels=dict(color="Percentage")
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Actionability analysis
    st.markdown("#### 🔄 Actionability Analysis")
    
    actionable_analysis = df.groupby(['Category', 'Actionable']).size().unstack(fill_value=0)
    actionable_analysis['Actionable_Rate'] = (
        actionable_analysis['Yes'] / 
        (actionable_analysis['Yes'] + actionable_analysis['No']) * 100
    ).round(2)
    
    fig_actionable = px.bar(
        x=actionable_analysis.index,
        y=actionable_analysis['Actionable_Rate'],
        color=actionable_analysis['Actionable_Rate'],
        color_continuous_scale='viridis',
        title="Actionability Rate by Category (%)"
    )
    fig_actionable.update_xaxes(tickangle=45)
    st.plotly_chart(fig_actionable, use_container_width=True)

# === RUN APP ===
if __name__ == "__main__":
    main()