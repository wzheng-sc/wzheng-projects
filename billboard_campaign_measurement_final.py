from datetime import date
from google.colab import auth, data_table, syntax, files
from google.cloud import bigquery
from IPython.display import display, Markdown
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings

auth.authenticate_user()


class BillboardCampaignMeasurement:
    def __init__(self, end_date, lookback_quarters):
        self.client = bigquery.Client(project='feelinsonice-hrd')
        self.end_date_dt = pd.to_datetime(end_date)
        self.quarter_formatted = f"{self.end_date_dt.year}Q{self.end_date_dt.quarter}"
        self.lookback_quarters = lookback_quarters
        self.start_date_dt = self.end_date_dt - DateOffset(months=3 * self.lookback_quarters)
        self.start_date_str = self.start_date_dt.strftime('%y%m%d')
        self.end_date_str = self.end_date_dt.strftime('%y%m%d')

        self.initialize_data()

    def initialize_data(self):
        self.run_campaign_queries()
        self.run_user_queries()
        self.run_multi_impression_queries()

    def run_campaign_queries(self):
        campaign_query = f'''
        SELECT *
        FROM `sc-analytics.report_growth.billboard_campaign_summary_quarterly_20*`
        WHERE 1=1
        AND _TABLE_SUFFIX BETWEEN '{self.start_date_str}' AND '{self.end_date_str}'
        AND campaign_id != ''
        '''
        self.df = self.run_query(campaign_query)
        self.df['quarter'] = pd.PeriodIndex(self.df['ts'], freq='Q').strftime('%YQ%q')
        self.df['month'] = pd.PeriodIndex(self.df['ts'], freq='M').strftime('%Y-%m')

    def run_user_queries(self):
        user_query = f"""
        SELECT *
        FROM `sc-analytics.report_growth.billboard_campaign_user_summary_quarterly_20*`
        WHERE _TABLE_SUFFIX BETWEEN '{self.start_date_str}' AND '{self.end_date_str}'
        """
        self.df_user = self.run_query(user_query)
    
    def run_multi_impression_queries(self):
      try:
        multi_impression_query = f"""
        select *
        from `sc-analytics.report_growth.billboard_campaign_impression_efficiency_quarterly_20*`
        where _TABLE_SUFFIX BETWEEN '{self.start_date_str}' AND '{self.end_date_str}'
        and quarter = '{self.quarter_formatted}'
        """
        self.df_multi_impression = self.run_query(multi_impression_query)

      except Exception as e:
        print(f"Failed to initialize multi impression data: {e}")

    def run_query(self, query):
        return self.client.query(query).to_dataframe()

    def validation_parameters(self):
        earliest_date = pd.to_datetime('2023-03-31')
        if self.start_date_dt < earliest_date:
            raise ValueError("The earliest data available is 2023Q1!")

        end_date_md = self.end_date_dt.strftime('%m%d')
        if end_date_md not in ['0331', '0630', '0930', '1231']:
            raise ValueError("The end date of the quarter is incorrect!")

    def user_distribution(self, var):
        selected_columns = ['quarter'] + [col for col in self.df_user.columns if var in col]
        df_selected = self.df_user[selected_columns]
        df_selected = df_selected.rename(columns=lambda x: x.replace(var+'_', ''))
        df_melted = pd.melt(df_selected, id_vars=['quarter'], var_name=var, value_name='value')
        df_pivot = df_melted.pivot(index=var, columns='quarter', values='value')
        df_pivot.columns.name = None

        return df_pivot.loc[['min', 'p25', 'median', 'p75']]

    def monthly_impression_change(self, df_monthly_impression):
        fig = go.Figure(go.Waterfall(
            name="MoM%",
            orientation="v",  # Vertical orientation
            measure=["relative"] * (len(df_monthly_impression) - 1),  
            x=df_monthly_impression['month'].astype(str),  # X-axis labels
            textposition="outside",
            y=df_monthly_impression['impression_count_mom%'].fillna(0).tolist()[1:],  #
            connector={"line":{"color":"rgb(63, 63, 63)"}},  # Line color between bars
            increasing={"marker":{"color":"green"}},  # Color for increase
            decreasing={"marker":{"color":"red"}},  # Color for decrease
            totals={"marker":{"color":"blue"}}  
        ))

        fig.update_layout(
            title="Month-over-Month Overall Impression Percentage Difference",
            showlegend=True,
            yaxis=dict(
                title='Percentage Difference',
                ticksuffix='%'
            )
        )
        fig.show()

    def visual_surface_impression_distribution(self, surface_merge_cnt):
        df_surface_impression = surface_merge_cnt.pivot_table(index='quarter', columns='surface', values='impression_distribution', fill_value=0)
        surface_new_order = ['FEED_HEADER_PROMPT', 'PROFILE_ACTIVITY_CARD', 'FULL_SCREEN_TAKEOVER']
        df_surface_impression = df_surface_impression[surface_new_order]

        colors = ['cornflowerblue', 'skyblue', 'yellowgreen']
        ax = df_surface_impression.plot(kind='barh', stacked=True, figsize=(9, 4), color = colors)
        plt.title('Distribution of Impressions by Quarter and Surface')
        plt.ylabel('Quarter')
        plt.xlabel('Percentage of Impressions (%)')
        plt.legend(title='Surface',  loc = 'upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        for bar in ax.patches:
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            x = bar.get_x() + width / 2
            label_text = f'{width:.1f}%'

            #only add text if there is enough space in the bar for it
            if width > 0:
                ax.text(x, y, label_text, ha='center', va='center', color='black')

        plt.show()

    def high_level_summary(self):
      # count the number of campaigns and impression by quarter
      campaign_cnt = self.df.groupby('quarter').agg({'campaign_id': 'nunique'})\
                        .rename(columns={'campaign_id': 'campaign_count'})
      impression_cnt = self.df[self.df['action']=='IMPRESSION'].groupby('quarter')\
                            .agg({'action_cnt': 'sum'}).rename(columns={'action_cnt': 'impression_count'})
      merge_cnt = campaign_cnt.merge(impression_cnt, on = 'quarter')
      df_summary = merge_cnt.reset_index()
      df_summary.sort_values('quarter', inplace=True)
      df_summary['campaign_count_qoq%'] = np.round(df_summary['campaign_count'].pct_change()*100,2)
      df_summary['impression_count_qoq%'] = np.round(df_summary['impression_count'].pct_change()*100,2)

      new_order = [
          'quarter',
          'campaign_count',
          'campaign_count_qoq%',
          'impression_count',
          'impression_count_qoq%'
      ]
      df_summary = df_summary[new_order].fillna(0) # return df_summary

      # campaigns in this quarter but not in previous x quarters
      this_quarter_campaigns = self.df[self.df['quarter'] == self.quarter_formatted]['campaign_id'].unique()
      this_quarter_campaigns = set([x.upper() for x in this_quarter_campaigns if isinstance(x, str)])

      previous_quarters_campaigns = self.df[self.df['quarter'] != self.quarter_formatted]['campaign_id'].unique()
      previous_quarters_campaigns = set([x.upper() for x in previous_quarters_campaigns if isinstance(x, str)])  # Convert to uppercase

      unique_to_this_quarter = this_quarter_campaigns.difference(previous_quarters_campaigns)

      
      # count by surface and quarter
      surface_campaign_cnt = self.df.groupby(['quarter', 'surface']).agg({'campaign_id': 'nunique'}).rename(columns={'campaign_id': 'surface_campaign_count'})
      surface_impression_cnt = self.df[self.df['action']=='IMPRESSION'].groupby(['quarter', 'surface']).agg({'action_cnt': 'sum'}).rename(columns={'action_cnt': 'surface_impression_count'})
      surface_merge_cnt = surface_campaign_cnt.merge(surface_impression_cnt, on = ['quarter', 'surface'])\
                                              .join(merge_cnt, on = ['quarter'])


      surface_merge_cnt['impression_distribution'] = np.round((surface_merge_cnt['surface_impression_count'] / surface_merge_cnt['impression_count']) * 100, 2)
      surface_merge_cnt['impression_billionth'] = np.round(surface_merge_cnt['surface_impression_count']/1000000000,2)
      df_surface_campaign_summary = surface_merge_cnt.pivot_table(index='quarter', columns='surface', values='surface_campaign_count', fill_value=0)\
                                                    .reset_index()
      df_surface_campaign_summary.columns.name = None
      df_surface_impression_summary =surface_merge_cnt.pivot_table(index='quarter', columns='surface', values='impression_billionth', fill_value=0)\
                                                    .reset_index()
      df_surface_impression_summary.columns.name = None
                                        
      campaign_count_value = df_summary[df_summary['quarter'] ==self.quarter_formatted]['campaign_count'].iloc[0]
      impression_count_value = df_summary[df_summary['quarter'] ==self.quarter_formatted]['impression_count'].iloc[0]
      fhp_campaign_count_value = df_surface_campaign_summary[df_surface_campaign_summary['quarter'] ==self.quarter_formatted]['FEED_HEADER_PROMPT'].iloc[0]
      fst_campaign_count_value = df_surface_campaign_summary[df_surface_campaign_summary['quarter'] ==self.quarter_formatted]['FULL_SCREEN_TAKEOVER'].iloc[0]
      pac_campaign_count_value = df_surface_campaign_summary[df_surface_campaign_summary['quarter'] ==self.quarter_formatted]['PROFILE_ACTIVITY_CARD'].iloc[0]


      # monthly campaign number and impression change 
      monthly_impression_cnt = self.df[self.df['action']=='IMPRESSION'].groupby('month')\
                                    .agg({'action_cnt': 'sum'}).rename(columns={'action_cnt': 'impression_count'})
      df_monthly_impression= monthly_impression_cnt.reset_index()
      df_monthly_impression.sort_values('month', inplace=True)
      df_monthly_impression['impression_count_mom%'] = np.round(df_monthly_impression['impression_count'].pct_change()*100,2)

      #user level
      user_campaign = self.user_distribution('campaign')
      user_impression = self.user_distribution('impression')

      print(f"• In {self.quarter_formatted}, we sent {campaign_count_value} billboard campaigns (with >20K impressions).")
      print(f"• {fhp_campaign_count_value} FHP, {fst_campaign_count_value} FST, and {pac_campaign_count_value} PAC.")
      print(f"• We received {impression_count_value/1000000000:.1f}B impression from these three surfaces.")
      print(f"• Here are the campaign ids unique in this quarter, and not seen in past {self.lookback_quarters} quarter(s):", unique_to_this_quarter)
      print("\n")

      display(Markdown("**Here is a high level summary:**"))
      print("\nTable 1: Overall Summary")
      display(df_summary)

      print("\nChart 1: Trend of Overall Impression Percentage Difference")
      self.monthly_impression_change(df_monthly_impression)
      display(df_monthly_impression)

      print("\nTable 2: Surface Summary - Campaign Count")
      display(df_surface_campaign_summary)

      print("\nTable 3: Surface Summary - Impression Count (Billionth)")
      display(df_surface_impression_summary)

      print("\nChart 2: Surface Summary - Impression Distribution")
      self.visual_surface_impression_distribution(surface_merge_cnt)

      print("\nTable 4: User Summary - Campaign")
      display(user_campaign)

      print("\nTable 5: User Summary - Impression")
      display(user_impression)


    def campaign_perf(self):
        #campaign QoQ performance
        ##impression, click and dismiss count
        df_quarter = self.df.groupby(['quarter','surface', 'action']).agg({'action_cnt':np.sum, 'campaign_id':'nunique'})
        df_quarter_pivot = df_quarter.pivot_table(index=['surface', 'action'], columns='quarter', values='action_cnt', fill_value=0)
        qoq_changes = df_quarter_pivot.pct_change(axis='columns') * 100
        for col in qoq_changes.columns[1:]:
            new_col_name = f'{col}_qoq_change(%)'
            df_quarter_pivot[new_col_name] = np.round(qoq_changes[col], 2)
        df_quarter_pivot.replace([np.inf, -np.inf, np.nan], 0, inplace=True)


        #conversion at overall and campaign level
        df_conversion = self.df.pivot_table(index=['surface', 'quarter'], columns='action', values='action_cnt', aggfunc='sum', fill_value=0)
        df_conversion['overall_ctr'] = np.round(df_conversion['CLICK'] / df_conversion['IMPRESSION'],4)
        df_conversion['overall_dismiss_rate'] = np.round(df_conversion['DISMISS'] / df_conversion['IMPRESSION'],4)
        df_conversion_campaign_level = self.df[self.df['quarter']== self.quarter_formatted].pivot_table(index=['surface', 'campaign_id'], columns='action', values='action_cnt', aggfunc='sum', fill_value=0)
        df_conversion_campaign_level['campaign_ctr'] = np.round(df_conversion_campaign_level['CLICK'] / df_conversion_campaign_level['IMPRESSION'],4)
        df_conversion_campaign_level['campaign_dismiss_rate'] = np.round(df_conversion_campaign_level['DISMISS'] / df_conversion_campaign_level['IMPRESSION'],4)
        df_conversion_subset = df_conversion[['overall_ctr', 'overall_dismiss_rate']].xs(self.quarter_formatted, level = 'quarter')
        df_conversion_campaign_merged = df_conversion_campaign_level.reset_index().merge(df_conversion_subset, on = 'surface')

        # construct FHP and FST dataframes for visualization and summary
        df_campaign_level_fhp = df_conversion_campaign_merged[df_conversion_campaign_merged['surface']=='FEED_HEADER_PROMPT'].drop(['CLICK','CLICK_EXTRA_BUTTON','DISMISS'], axis = 1)
        df_campaign_level_fst = df_conversion_campaign_merged[df_conversion_campaign_merged['surface']=='FULL_SCREEN_TAKEOVER'].drop(['CLICK','CLICK_EXTRA_BUTTON','DISMISS'], axis = 1)
        df_campaign_level_fhp.columns.name = None
        df_campaign_level_fst.columns.name = None

        # top 5 and bottom 5 FHP and FST campaigns
        top_5_fhp_campaign_ctr = df_campaign_level_fhp.nlargest(5, 'campaign_ctr')
        bottom_5_fhp_campaign_ctr = df_campaign_level_fhp.nsmallest(5, 'campaign_ctr')
        top_5_fst_campaign_ctr = df_campaign_level_fst.nlargest(5, 'campaign_ctr')
        bottom_5_fst_campaign_ctr = df_campaign_level_fst.nsmallest(5, 'campaign_ctr')


        print('Table 6: Impression, Click and Dismiss Summary')
        display(df_quarter_pivot)

        # print('\nChart 3: FHP Campaign Performance')
        # self.visual_campaign_conversion(df_campaign_level_fhp, 'FHP')

        # print('\nChart 4: FST Campaign Performance')
        # self.visual_campaign_conversion(df_campaign_level_fst, 'FST')

        print('\nTable 7: Top 5 FHP campaigns (by CTR)')
        display(top_5_fhp_campaign_ctr)

        print('\nTable 8: Bottom 5 FHP campaigns (by CTR)')
        display(bottom_5_fhp_campaign_ctr)

        print('\nTable 9: Top 5 FST campaigns (by CTR)')
        display(top_5_fst_campaign_ctr)

        print('\nTable 8: Bottom 5 FST campaigns (by CTR)')
        display(bottom_5_fst_campaign_ctr)

        df_conversion_campaign_merged.to_csv(f'billboard_measurement_{self.quarter_formatted}.csv', index = False)
        files.download(f'billboard_measurement_{self.quarter_formatted}.csv')
        print(f"\n ** A complete list of campaigns is downloaded. Check file 'billboard_measurement_{self.quarter_formatted}.csv' in your Download folder!")


    def visual_campaign_conversion(self, df_selected, surface):
        df_selected.plot.scatter('campaign_ctr', 'campaign_dismiss_rate', s=df_selected['IMPRESSION'].astype(np.float64)/1000000, alpha=0.5)
        # Adding titles and labels
        plt.title(f"{surface} - Campaign Performance")
        plt.xlabel("Conversion Rate")
        plt.ylabel("Dismiss Rate")

        plt.show()

    def campaign_impression_efficiency(self, campaign_id):
        # Suppress all warnings
        warnings.filterwarnings('ignore')
        
        df_multi_impression_selected = self.df_multi_impression.loc[(self.df_multi_impression['uu'] >100) & 
                                              (self.df_multi_impression['campaign_id'] == campaign_id) &
                                              (self.df_multi_impression['impression'] > 0) & 
                                              (self.df_multi_impression['impression'] <16)]
        df_multi_impression_selected['total_impression'] = df_multi_impression_selected['uu'] * df_multi_impression_selected['impression']
        df_multi_impression_selected['ctr'] = np.round((df_multi_impression_selected['total_click'] / df_multi_impression_selected['total_impression']),4)
        df_multi_impression_selected['dismiss_rate'] = np.round((df_multi_impression_selected['total_dismiss'] / df_multi_impression_selected['total_impression']),4)

        df_multi_impression_selected['uu_million'] = df_multi_impression_selected['uu']/1000000
        df_multi_impression_selected['ctr%'] = df_multi_impression_selected['ctr'] *100
        df_multi_impression_selected['dismiss_rate%'] = df_multi_impression_selected['dismiss_rate'] *100
        # df_multi_impression_selected = df_multi_impression_selected.loc[(df_multi_impression_selected['impression'] > 0) & (df_multi_impression_selected['impression'] <16)]
        df_multi_impression_selected['impression'] = df_multi_impression_selected['impression'].astype(int)

        # Sort 'impression' numerically
        df_multi_impression_selected = df_multi_impression_selected.sort_values('impression')
        df_multi_impression_selected['impression_str'] = df_multi_impression_selected['impression'].astype(str)

        fig, ax1 = plt.subplots(ncols=1, figsize=(8, 4))

        sns.barplot(data=df_multi_impression_selected, x='impression_str', y='uu_million', color='lightblue', ax=ax1, errorbar = None)
        ax1.set_title(f"{campaign_id}")
        ax1.set_ylabel('UU in Million')
        ax1.set_xlabel('#Impression Received')

        ax2 = ax1.twinx()
        ax2.grid(False)
        sns.lineplot(data=df_multi_impression_selected, x='impression_str', y='ctr%', marker='o', color='green', label = 'CTR', ax=ax2, errorbar = None)
        sns.lineplot(data=df_multi_impression_selected, x='impression_str', y='dismiss_rate%', marker='s', color='red', label = 'Dismiss Rate', ax=ax2, errorbar = None)
        ax2.set_ylabel('Rate(%)', color='red')
        ax2.legend(title='Measure')


        plt.tight_layout()
        plt.show()

        display(df_multi_impression_selected[['quarter', 'impression', 'uu', 'total_click', 'total_dismiss', 'total_impression',
                                    'ctr%', 'dismiss_rate%']])



      
