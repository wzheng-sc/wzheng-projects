#Import relevant package and connect to the BQ server, please don't edit the cell
#from google.colab import auth, data_table, syntax
from google.cloud import bigquery
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import pytz

auth.authenticate_user()

class CampaignTarget:
    def __init__(self, table_name, campaign_type, is_community_campaign , country_single, country_multiple, l7_from, l7_to, l90_from, l90_to, locale, minimum_app_version):
        # Time element
        self.pstTz = pytz.timezone("America/Los_Angeles")
        self.run_datetime = datetime.now(self.pstTz) - timedelta(days=1)
        self.formatted_run_date = self.run_datetime.date().strftime('%Y%m%d')


        # Campaign parameters
        self.table_name = table_name
        self.campaign_type = campaign_type
        self.is_community_campaign = is_community_campaign
        self.country_single = country_single
        self.country_multiple = country_multiple
        self.l7_from = l7_from
        self.l7_to = l7_to
        self.l90_from = l90_from
        self.l90_to = l90_to
        self.locale = locale
        self.minimum_app_version = minimum_app_version
        
        
    def validation(self):
        country_input = self.transform_country_input()
        self.validation_parameters()
        print(f' • We are creating the targeting with the following conditions:\n'
              f' • Campaigm type: {self.campaign_type}\n'
              f' • Community campaign: {self.is_community_campaign}\n'
              f' • Country: {country_input}\n'
              f' • l7: [{self.l7_from}, {self.l7_to}] *inclusive\n'
              f' • l90: [{self.l90_from}, {self.l90_to}] *inclusive\n'
              f' • locale: {self.locale}\n'
              f' • App version: at least {self.minimum_app_version} or above\n')

    def validation_parameters(self):
        if self.country_single and self.country_multiple:
            raise ValueError('Error: Cannot specify both single and multiple country inputs.')
        if self.l7_from > self.l7_to:
            raise ValueError('Error: l7 start date must be before or on the same day as end date.')
        if self.l90_from > self.l90_to:
            raise ValueError('Error: l90 start date must be before or on the same day as end date.')

    def transform_country_input(self):
        country_input = ''
        if self.country_multiple == '' and self.country_single == 'All':
            all_country_query = """
                SELECT DISTINCT country AS country_code
                FROM `sc-analytics.report_app.country_mapping`
                WHERE name != ''
                      AND UPPER(name) NOT LIKE 'UNKNOWN%'
            """
            try:
                query_job = self.client.query(all_country_query)
                result = query_job.result()
                country_input = "', '".join([row.country_code for row in result])
            except Exception as e:
                print("Failed to fetch data: ", e)

        elif self.country_multiple == '':
            country_input = self.country_single
        else:
            country_input = self.country_multiple
        return country_input


    def query_generator(self):

        broadcast_attributes = "map.user_id AS userID"
        broadcast_mapping_table = """JOIN `sc-mjolnir.enigma.user_map_20*` AS map
            ON map.ghost_id = dau.ghost_user_id"""
        broadcast_mapping_table_condition = """AND map._TABLE_SUFFIX = FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY))"""

        email_attributes = """id.ghost_user_id
            , CASE WHEN
              MOD(ABS(FARM_FINGERPRINT(CAST(ghost_user_id AS STRING))), 10) < 5 THEN 'treatment'
              ELSE 'control' END AS group_assignment
            , DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY) AS run_date"""
        email_bounce_cte =  """WITH email_bounce AS
            (
            SELECT
            ghost_user_id
            FROM `sc-portal.quest.notif_email_campaign_user_20*`
            WHERE
            (_TABLE_SUFFIX
            BETWEEN FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 YEAR))
            AND FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY)))
            AND(
            email_campaign_bounce>0
            OR email_campaign_unsubscribe>0
            OR email_transactional_bounce>0
            OR email_transactional_unsubscribe>0
            )
            )"""
        email_bounce_table = "LEFT JOIN email_bounce AS e USING (ghost_user_id)"
        email_bounce_condition = "AND e.ghost_user_id IS NULL"
        email_verified_condition = "AND id.isemailverified"
        user_cohort_table = "JOIN `sc-analytics.report_search.user_cohorts_20*` AS uc USING (ghost_user_id)"
        user_cohort_table_condition = """AND uc._TABLE_SUFFIX = FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY))"""
        last_activity_table = "JOIN `sc-analytics.report_app.last_active_day_20*` AS la USING (ghost_user_id)"
        last_activity_table_condition = """AND la._TABLE_SUFFIX = FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY))"""
        locale_condition = f"""AND uc.locale = '{self.locale}' """
        l_7_condition = f"AND la.l_7 BETWEEN {self.l7_from} AND {self.l7_to}"
        l_90_condition = f"AND la.l_90 BETWEEN {self.l90_from} AND {self.l90_to}"
        app_condition = f"""AND (CAST(SPLIT(id.version, '.')[SAFE_OFFSET(0)] AS INT)*1e2 + CAST(SPLIT(id.version, '.')[SAFE_OFFSET(1)] AS INT))/100.0 >= {self.minimum_app_version}"""
        country_input = self.transform_country_input()
        table, table_with_suffix, table_view = self.table_config()

        # Campaign type-specific configurations
        if self.campaign_type == "broadcast":
            email_attributes = email_bounce_cte = email_bounce_table \
                = email_bounce_condition = email_verified_condition = ""
        #     table_project = "sc-notif-campaigns"
        #     table_prefix = "scheduled_broadcast_targeting"
        #     table = '.'.join([table_project, table_prefix, self.table_name])
        elif self.campaign_type == "email":
            broadcast_attributes = broadcast_mapping_table \
                = broadcast_mapping_table_condition = ""
            # table_project = "email-infra-prod"
            # table_prefix = "bigquery_targeting_campaigns"
            # table = f"sc-product-datascience.wzheng.{self.table_name}"
            # table_view = '.'.join([table_project, table_prefix, self.table_name])

        # Locale specific conditions
        if self.locale is None:
            user_cohort_table = user_cohort_table_condition = locale_condition = ""

        # Date specific conditions
        if self.l7_from == 0 and self.l7_to == 7 and self.l90_from == 0 and self.l90_to == 90:
            last_activity_table = last_activity_table_condition = l_7_condition = l_90_condition = ""
        elif self.l7_from == 0 and self.l7_to == 7:
            l_7_condition = ""
        elif self.l90_from == 0 and self.l90_to == 90:
            l_90_condition = ""

        # App version specific conditions
        if self.minimum_app_version == 0:
            app_condition = ""

        main_query = f"""

            {email_bounce_cte}
            SELECT DISTINCT
            {email_attributes}
            {broadcast_attributes}

            FROM `sc-analytics.report_app.dau_user_country_20*` AS dau

            #for verification info
            JOIN  `sc-analytics.report_user.identity_20*` AS id
            USING (ghost_user_id)

            {last_activity_table}  ##join the table when need L7 or L90
            {user_cohort_table} ##join cohort table when need locale

            {broadcast_mapping_table}
            {email_bounce_table}

            WHERE
            1 = 1
            AND dau.country IN ('{country_input}')

            AND dau._TABLE_SUFFIX = FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY))
            AND id._TABLE_SUFFIX = FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY))
            AND la._TABLE_SUFFIX = FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY))
            {broadcast_mapping_table_condition}
            {email_bounce_condition}
            {email_verified_condition}
            {user_cohort_table_condition}
            {l_7_condition}
            {l_90_condition}
            {app_condition}
        """
        runnable_query = f"""
            ## Create dynamic suffix

            DECLARE suffix STRING DEFAULT FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE('America/Los_Angeles'), INTERVAL 2 DAY));

            EXECUTE IMMEDIATE \"\"\"
            CREATE OR REPLACE TABLE `{table}_\"\"\" || suffix || \"\"\"` AS
            {main_query}
            \"\"\";

        """
        email_query_view =  f"""
            CREATE OR REPLACE VIEW
            {table_view}
            OPTIONS (expiration_timestamp=TIMESTAMP "2999-01-01 00:00:00") AS

            SELECT DISTINCT
            ghost_user_id
            FROM `{table}_20*`
            WHERE
            1=1
            AND run_date = DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY)
            AND group_assignment = 'treatment';

        """


        return main_query, runnable_query, email_query_view

    def table_config(self):

        if self.campaign_type == "broadcast":
              table_project = "sc-notif-campaigns"
              table_prefix = "scheduled_broadcast_targeting"
              table = '.'.join([table_project, table_prefix, self.table_name])
              table_with_suffix = f"""{table}_{self.formatted_run_date}"""
              table_view = None

        elif self.campaign_type == "email":
              table_project = "email-infra-prod"
              table_prefix = "bigquery_targeting_campaigns"
              table = f"sc-product-datascience.wzheng.{self.table_name}"
              table_with_suffix = None
              table_view = '.'.join([table_project, table_prefix, self.table_name])

        return table, table_with_suffix, table_view


    def display_query(self):
        main_query, runnable_query, email_query_view = self.query_generator()
        print("The query for the {} campaign has been generated.\n"\
                .format(self.campaign_type))
        if self.campaign_type =='broadcast':
            print(runnable_query)

        elif self.campaign_type =='email':
            print(runnable_query, '\n')
            print(email_query_view)


    def execute_query(self):
        main_query, runnable_query, email_query_view = self.query_generator()
        table, table_with_suffix, table_view = self.table_config()
        client = bigquery.Client(project="feelinsonice-hrd")

        job_config = bigquery.QueryJobConfig(destination = table_with_suffix\
                                           , write_disposition="WRITE_TRUNCATE")
        query_job = client.query(main_query, job_config=job_config)

        if self.campaign_type == 'broadcast':
          try:
              print("Executing...")
              query_job.result()  # Wait for the job to complete.
              count_query = f"""
              SELECT COUNT(1) AS total_rows 
              FROM {table_with_suffix} 
              """
              print(f"Table {table_with_suffix} created successfully.")

          except Exception as e:
              print(f"Failed to create table: {e}")

        elif self.campaign_type == 'email':
            try:
              print("Executing...")
              query_job.result()  # Wait for the job to complete.
              bq_view = bigquery.Table(table_view)
              view_query = f"""
                CREATE OR REPLACE VIEW
                {table_view}
                OPTIONS (expiration_timestamp=TIMESTAMP "2999-01-01 00:00:00") AS

                SELECT DISTINCT
                ghost_user_id
                FROM `{table}_20*`
                WHERE
                1=1
                AND run_date = DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 1 DAY)
                AND group_assignment = 'treatment';"""

              view_job = client.query(view_query)
              view_job.result()
              count_query = f"""
              SELECT COUNT(1) AS total_rows 
              FROM {table_view}
              """
              print(f"Successfully created view at {table_view}")
              
            except Exception as e:
                  print(f"Failed to create view: {e}")

    def target_count(self):
        table, table_with_suffix, table_view = self.table_config()
        client = bigquery.Client(project="feelinsonice-hrd")

        if self.campaign_type == 'broadcast':
            try:
              print("Counting...")
              count_query = f"""
              SELECT COUNT(1) AS total_rows 
              FROM {table_with_suffix} 
              """
              count_job = client.query(count_query)
              count_result = count_job.result() 

              # Retrieve and print the count result
              for row in count_result:
                print(f"Total number of targeting: {row.total_rows:,}")

            except Exception as e:
              print(f"Failed to count the targeting: {e}")

        elif self.campaign_type == 'email':
            try:
              print("Counting...")
              count_query = f"""
              SELECT COUNT(1) AS total_rows 
              FROM {table_view}
              """
              count_job = client.query(count_query)
              count_result = count_job.result()  # Waits for the job to complete
              
              # # Retrieve and print the count result
              for row in count_result:
                 print(f"Total number of targeting: {row.total_rows:,}")

            except Exception as e:
              print(f"Failed to count the targeting: {e}")

