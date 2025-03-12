#Import relevant package and connect to the BQ server, please don't edit the cell
#from google.colab import auth, data_table, syntax
from google.cloud import bigquery
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import pytz
import re

auth.authenticate_user()

class CampaignTarget:
    def __init__(
    self,
    table_name: str,
    table_to_delete: str,
    campaign_type: str,
    country_single: str,
    country_multiple: str,
    l7_sign: str,
    l7_value: int,
    l90_sign: str,
    l90_value: int,
    locale: str,
    minimum_app_version: bool,
    minimum_app_version_value: float,
    device_type: str,
    has_age_condition: bool,
    age_from: int,
    age_to: int,
    is_community_campaign: bool,
    is_community_member: bool,
    creator_tier: str,
    is_snapchatplus_campaign: bool,
    is_snapchatplus_subscriber: bool,

):
        # Time element
        self.pstTz = pytz.timezone("America/Los_Angeles")
        self.run_datetime = datetime.now(self.pstTz) - timedelta(days=2)
        self.formatted_run_date = self.run_datetime.date().strftime('%Y%m%d')


        # Campaign parameters
        self.table_name = table_name
        self.table_to_delete = table_to_delete
        self.campaign_type = campaign_type
        self.country_single = country_single
        self.country_multiple = country_multiple
        self.l7_sign = l7_sign
        self.l7_value = l7_value
        self.l90_sign = l90_sign
        self.l90_value = l90_value
        self.locale = locale
        self.minimum_app_version = minimum_app_version
        self.minimum_app_version_value = minimum_app_version_value
        self.device_type = device_type
        self.has_age_condition = has_age_condition
        self.age_from = age_from
        self.age_to = age_to
        self.is_community_campaign = is_community_campaign
        self.is_community_member = is_community_member
        self.creator_tier = creator_tier 
        self.is_snapchatplus_campaign = is_snapchatplus_campaign
        self.is_snapchatplus_subscriber = is_snapchatplus_subscriber


    def validation(self):
        country_input = self.transform_country_input()
        creator_tier_input = self.transform_creator_tier_input()
        self.validation_parameters()
        print(f'We are creating the targeting with the following conditions:'
              f'\n'
              f' • Campaigm type: {self.campaign_type}\n'
              f""" • Country:'{country_input}'\n"""
              f' • Locale: {self.locale}\n'
              f""" • Creator Tier: '{creator_tier_input}'""")
        #L7
        if self.l7_sign == "not required":
          print(f' • L7: {self.l7_sign}')
        else:
          print(f' • L7: {self.l7_sign}{self.l7_value}')

        #L90
        if self.l90_sign == "not required":
          print(f' • L90: {self.l90_sign}')
        else:
          print(f' • L90: {self.l90_sign}{self.l90_value}')

        #app version
        if not self.minimum_app_version:
          print(f' • Minimum app version: not required')
        else:
          print(f' • Minimum app version: >= {self.minimum_app_version_value}')

        #device type 
        print(f' • Device Type: {self.device_type}')

        #age
        if not self.has_age_condition:
          print(f' • Age: not required')
        else:
          print(f' • Age: between {self.age_from} and {self.age_to} *inclusive')

        #community
        if not self.is_community_campaign:
          print(f' • Community campaign: {self.is_community_campaign}')
        elif self.is_community_campaign and self.is_community_member:
          print(f' • Community campaign: {self.is_community_campaign}, targeted Community Member')
        else:
          print(f' • Community campaign: {self.is_community_campaign}, targeted Non Community Member')

        #snapchat plus
        if not self.is_snapchatplus_campaign:
          print(f' • Snapchat+ campaign: {self.is_snapchatplus_campaign}')
        elif self.is_snapchatplus_campaign and self.is_snapchatplus_subscriber:
          print(f' • Snapchat+ campaign: {self.is_snapchatplus_campaign}, targeted Snapchat+ subscribers')
        else:
          print(f' • Snapchat+ campaign: {self.is_snapchatplus_campaign}, targeted Non Snapchat+ subscribers')




    def validation_parameters(self):
        if self.country_single and self.country_multiple:
            raise ValueError("Error: Cannot specify both single and multiple country inputs.")
        if self.age_from > self.age_to:
            raise ValueError("Error: age's upper bound is small than the lower bound.")


    def transform_country_input(self):
        client = bigquery.Client(project="feelinsonice-hrd")
        country_input = ''
        if self.country_multiple == '' and self.country_single == 'All':
            all_country_query = """
                SELECT DISTINCT country AS country_code
                FROM `sc-analytics.report_app.country_mapping`
                WHERE name != ''
                      AND UPPER(name) NOT LIKE 'UNKNOWN%'
            """
            try:
                query_job = client.query(all_country_query)
                result = query_job.result()
                country_input = "', '".join([row.country_code for row in result])
            except Exception as e:
                print("Failed to fetch data: ", e)

        elif self.country_multiple == '':
            country_input = self.country_single
        else:
            country_codes = self.country_multiple.split(", ")
            formatted_string = "', '".join(country_codes)
            country_input = formatted_string
            
        return country_input
        
    def transform_creator_tier_input(self):
        creator_tier_input = ''

        if self.creator_tier == "Beginner":
            creator_tier_input = "2 - Beginner"
        
        elif self.creator_tier == "Beginner+":
            creator_tier_input = "', '".join(["2 - Beginner", "3 - Elementary", "4 - Intermediate", "5 - Advanced", "6 - Expert"])

        elif self.creator_tier == "Elementary+":
            creator_tier_input = "', '".join(["3 - Elementary", "4 - Intermediate", "5 - Advanced", "6 - Expert"])

        elif self.creator_tier == "Intermediate+":
            creator_tier_input = "', '".join(["4 - Intermediate", "5 - Advanced", "6 - Expert"])
        
        elif self.creator_tier == "Advanced+":
            creator_tier_input = "', '".join(["5 - Advanced", "6 - Expert"])
        
        elif self.creator_tier == "Expert":
            creator_tier_input = "6 - Expert"

        elif self.creator_tier =="not required":
            creator_tier_input = "not required"

        return creator_tier_input

    def query_generator(self):

        broadcast_attributes = "map.user_id AS userID"
        broadcast_mapping_table = """JOIN `sc-mjolnir.enigma.user_map_v2` AS map
            ON map.ghost_id = uc.ghost_user_id"""

        email_attributes = """id.ghost_user_id"""
            # , CASE WHEN
            #   MOD(ABS(FARM_FINGERPRINT(CAST(ghost_user_id AS STRING))), 10) < 5 THEN 'treatment'
            #   ELSE 'control' END AS group_assignment
            # , DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY) AS run_date"""
        # email_bounce_cte =  """WITH email_bounce AS
        #     (
        #     SELECT
        #     ghost_user_id
        #     FROM `sc-portal.quest.notif_email_campaign_user_20*`
        #     WHERE
        #     (_TABLE_SUFFIX
        #     BETWEEN FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 YEAR))
        #     AND FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY)))
        #     AND(
        #     email_campaign_bounce>0
        #     OR email_campaign_unsubscribe>0
        #     OR email_transactional_bounce>0
        #     OR email_transactional_unsubscribe>0
        #     )
        #     )"""
        # email_bounce_table = "LEFT JOIN email_bounce AS e USING (ghost_user_id)"
        # email_bounce_condition = "AND e.ghost_user_id IS NULL"
        email_verified_condition = "AND id.isemailverified"
        identity_table = "JOIN  `sc-analytics.report_user.identity_20*` AS id USING (ghost_user_id)"
        identity_table_condition = """AND id._TABLE_SUFFIX = FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY))"""
        last_activity_table = "JOIN `sc-analytics.report_app.last_active_day_20*` AS la USING (ghost_user_id)"
        last_activity_table_condition = """AND la._TABLE_SUFFIX = FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY))"""
        community_table = "LEFT JOIN `sc-analytics.report_growth.community_custom_group_staging_20*` AS community ON community.ghost_user_id = uc.ghost_user_id"
        community_table_condition = """AND community._TABLE_SUFFIX = FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY))"""


        locale_condition = f"""AND LEFT(UPPER(uc.locale),2) = 'EN' """ if self.locale == "EN" else f"""AND LEFT(UPPER(uc.locale),2)!= 'EN'"""
        l_7_condition = f"AND la.l_7 {self.l7_sign}{self.l7_value}"
        l_90_condition = f"AND la.l_90 {self.l90_sign}{self.l90_value}"
        app_condition = f"""AND (CAST(SPLIT(id.version, '.')[SAFE_OFFSET(0)] AS INT)*1e2 + CAST(SPLIT(id.version, '.')[SAFE_OFFSET(1)] AS INT))/100.0 >= {self.minimum_app_version_value}"""
        device_type_condition = f"""AND UPPER(id.deviceType) = UPPER('{self.device_type}')"""
        age_condition = f"""AND id.age between {self.age_from} and {self.age_to}"""
        community_condition = f"""AND community.ghost_user_id IS NOT NULL""" if self.is_community_member else f"""AND community.ghost_user_id IS NULL"""
        #creator tier
        creator_tier_input = self.transform_creator_tier_input()
        creator_tier_condition = f"""AND uc.creator_tier IN ('{creator_tier_input}') """
        #snapchat plus condition
        snapchatplus_condition = f"AND uc.with_snapchat_plus= {self.is_snapchatplus_subscriber}"


        country_input = self.transform_country_input()
        table, table_with_suffix, table_view = self.table_config()

        # Campaign type-specific configurations
        if self.campaign_type == "broadcast- one time" or self.campaign_type == "broadcast- recurring" :
            # email_attributes = email_bounce_cte = email_bounce_table \
            #     = email_bounce_condition = \
            email_attributes = email_verified_condition = ""

        elif self.campaign_type == "email":
            broadcast_attributes = broadcast_mapping_table = ""

        # Community campaign configurations
        if not self.is_community_campaign:
          community_table = community_table_condition = community_condition = ""

        # Locale specific conditions
        if self.locale == "not required":
           locale_condition = ""

        # Date specific conditions
        if self.l7_sign == "not required" and self.l90_sign == "not required":
            last_activity_table = last_activity_table_condition = l_7_condition = l_90_condition = ""
        elif self.l7_sign == "not required" :
            l_7_condition = ""
        elif self.l90_sign == "not required":
            l_90_condition = ""

        # App version specific conditions
        if not self.minimum_app_version:
            app_condition = ""

        # Device type condition 
        if self.device_type == "not required":
          device_type_condition = ""

        # Age condition
        if not self.has_age_condition:
          age_condition = ""

        # Creator tier condition 
        if self.creator_tier == "not required":
          creator_tier_condition = ""


        # Identity table
        if self.minimum_app_version == False and self.device_type == "not required" and self.campaign_type != "email" and self.has_age_condition == False:
           identity_table = identity_table_condition = ""


        # main_query = f"""

        #     {email_bounce_cte}
        #     SELECT DISTINCT
        #     {email_attributes}
        #     {broadcast_attributes}

        #     FROM `sc-analytics.report_search.user_cohorts_20*` AS uc
        #     {broadcast_mapping_table}
        #     {identity_table}
        #     {last_activity_table}
        #     {community_table}
        #     {community_table_condition}
        #     {email_bounce_table}
        #     WHERE
        #     1 = 1
        #     AND uc._TABLE_SUFFIX = FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY))
        #     {identity_table_condition}
        #     {last_activity_table_condition}
        #     {community_condition}
        #     {email_bounce_condition}
        #     {email_verified_condition}
        #     {l_7_condition}
        #     {l_90_condition}
        #     {app_condition}
        #     {device_type_condition}
        #     {locale_condition}
        #     {age_condition}
        #     {creator_tier_condition}
        #     AND uc.l_90_country IN ('{country_input}')

        # """
        main_query = f"""

      
            SELECT DISTINCT
            {broadcast_attributes}
            {email_attributes}
            FROM `sc-analytics.report_search.user_cohorts_20*` AS uc
            {broadcast_mapping_table}
            {identity_table}
            {last_activity_table}
            {community_table}
            {community_table_condition}
            WHERE
            1 = 1
            AND uc._TABLE_SUFFIX = FORMAT_DATE("%y%m%d",DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY))
            {identity_table_condition}
            {last_activity_table_condition}
            {community_condition}
            {email_verified_condition}
            {l_7_condition}
            {l_90_condition}
            {app_condition}
            {device_type_condition}
            {locale_condition}
            {age_condition}
            {creator_tier_condition}
            {snapchatplus_condition}
            AND uc.l_90_country IN ('{country_input}')

        """
        if self.campaign_type == "broadcast- one time" or self.campaign_type == 'email':
            runnable_query = f"""
            ## Create dynamic suffix
            DECLARE suffix STRING DEFAULT FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE('America/Los_Angeles'), INTERVAL 2 DAY));

            EXECUTE IMMEDIATE \"\"\"
            CREATE OR REPLACE TABLE `{table}_\"\"\" || suffix || \"\"\"` AS
            {main_query}
            \"\"\";
        """
        elif self.campaign_type == "broadcast- recurring" :
            runnable_query = f"""
            CREATE OR REPLACE TABLE `{table}` AS {main_query}
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
            AND run_date = DATE_SUB(CURRENT_DATE('America/Los_Angeles'),INTERVAL 2 DAY);

        """


        return main_query, runnable_query, email_query_view

    def table_config(self):
        #broadcast- one time: has date suffix
        if self.campaign_type == "broadcast- one time":
              table_project = "sc-notif-campaigns"
              table_prefix = "scheduled_broadcast_targeting"
              table = '.'.join([table_project, table_prefix, self.table_name])
              table_with_suffix = f"""{table}_{self.formatted_run_date}"""
              table_view = None
        #broadcast- recurring: no date suffix
        elif self.campaign_type == "broadcast- recurring":
              table_project = "sc-notif-campaigns"
              table_prefix = "scheduled_broadcast_targeting"
              table = '.'.join([table_project, table_prefix, self.table_name])
              table_with_suffix = f"""{table}"""
              table_view = None

        elif self.campaign_type == "email":
              table_project = "email-infra-prod"
              table_prefix = "bigquery_targeting_campaigns"
              table = f"email-infra-prod.bigquery_targeting_campaigns.{self.table_name}"
              table_with_suffix = f"""{table}_{self.formatted_run_date}"""
              table_view = '.'.join([table_project, table_prefix, self.table_name])

        return table, table_with_suffix, table_view


    def display_query(self):
        main_query, runnable_query, email_query_view = self.query_generator()
        print("# The query for the {} campaign has been generated.\n"\
                .format(self.campaign_type))
        lines = runnable_query.split('\n')
        filtered_lines = [line for line in lines if line.strip()]
        formatted_runnable_query = '\n'.join(filtered_lines)

    # line is empty (has only the following: \t\n\r and whitespace)
        if self.campaign_type =="broadcast- one time" or self.campaign_type =="broadcast- recurring":
            print(formatted_runnable_query)

        elif self.campaign_type =='email':
            print(formatted_runnable_query, '\n')
            print(email_query_view)


    def execute_query(self):
        main_query, runnable_query, email_query_view = self.query_generator()
        table, table_with_suffix, table_view = self.table_config()
        client = bigquery.Client(project="feelinsonice-hrd")

        job_config = bigquery.QueryJobConfig(destination = table_with_suffix\
                                           , write_disposition="WRITE_TRUNCATE")
        query_job = client.query(main_query, job_config=job_config)

        if self.campaign_type =="broadcast- one time" or self.campaign_type =="broadcast- recurring":
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

            except Exception as e:
              print(f"Failed to create the table: {e}")

            try:
               print("Creating email view...")

               bq_view = bigquery.Table(table_view)
               view_query = f"""
                    CREATE OR REPLACE VIEW
                    {table_view}
                    OPTIONS (expiration_timestamp=TIMESTAMP "2999-01-01 00:00:00") AS

                    SELECT DISTINCT
                    ghost_user_id
                    FROM `{table}_20*`;
                    """

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

        if self.campaign_type =="broadcast- one time" or self.campaign_type =="broadcast- recurring":
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

    def delete_table(self):
       client = bigquery.Client(project="feelinsonice-hrd")
       try:
         print(f"Deleting table `{self.table_to_delete}`...")
         delete_query = f"""
         DROP TABLE `{self.table_to_delete}`

         """
         client.query(delete_query)
         print(f"Deleted successfully!")

       except Exception as e:
         print(f"Failed to delete table {self.table_to_delete}: {e}")
