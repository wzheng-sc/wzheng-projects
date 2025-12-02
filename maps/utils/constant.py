from utils.helper import majority_vote, combine_text_list

"""
Stores queries for the maps viral places.
TODO: This file should be removed when onboarding to flowrida.
"""

def get_viral_places_query(start_date: str, 
                           end_date: str,
                           view_weight: float,
                           freshness_weight: float,):
    return f"""
            WITH urls AS
            (SELECT story_snap_id, 
            MAX(TIMESTAMP_MILLIS(CAST(submit_ts AS INT64))) AS submit_ts,
            MAX(IF(unencrypted_flat_video_result_url != '', unencrypted_flat_video_result_url, media_url)) AS media_url
            FROM `context-pii.snapjoin.our_story_snap_2025*`
            WHERE _TABLE_SUFFIX BETWEEN RIGHT(FORMAT_DATE('%Y%m%d', DATE_SUB(PARSE_DATE('%Y%m%d', '{start_date}'), INTERVAL 1 DAY)), 4) AND RIGHT('{end_date}', 4)
            GROUP BY 1),

            viral_places_staging AS 
            (
            SELECT   
            continent, 
            inclusive_region, 
            place_country_code, 
            place_id,
            place_name, 
            place_region, 
            place_country, 
            MIN(event_time) AS detection_start_time,
            MAX(event_time) AS detection_end_time,
            # FROM `sc-analytics.report_maps.maps_ttp_demand_detection_result_locality_stories_*`
            FROM `sc-product-datascience.wzheng.maps_viral_place_locality_stories_*`
            WHERE 1=1
            AND DATE(event_time) BETWEEN PARSE_DATE('%Y%m%d', '{start_date}') AND PARSE_DATE('%Y%m%d', '{end_date}')
            GROUP BY ALL
            ),
            stories AS 
            (SELECT  
            stories.continent, 
            stories.inclusive_region, 
            stories.place_country_code, 
            stories.place_id,
            stories.place_name, 
            stories.place_region, 
            stories.place_country, 
            stories.story_snap_id, 
            stories.story_link, 
            stories.venue_name,
            MIN(viral_places_staging.detection_start_time) AS detection_start_time,
            MAX(viral_places_staging.detection_end_time) AS detection_end_time,
            MAX(stories.time_viewed_total_day) AS time_viewed_total_day, 
            MAX(stories.heatmap_story_views_total_day) AS heatmap_story_views_total_day,
            # FROM `sc-analytics.report_maps.maps_ttp_demand_detection_result_locality_stories_*` AS stories
            FROM `sc-product-datascience.wzheng.maps_viral_place_locality_stories_*` AS stories
            LEFT JOIN viral_places_staging 
            USING (place_id)
            LEFT JOIN urls 
            USING(story_snap_id) 
            WHERE 1=1
            AND DATE(event_time) BETWEEN PARSE_DATE('%Y%m%d', '{start_date}') AND PARSE_DATE('%Y%m%d', '{end_date}')
            AND urls.media_url != '' 
            AND media_url is not NULL
            GROUP BY ALL),
            viral_places_stories AS
            (
            SELECT  
            stories.continent, 
            stories.inclusive_region, 
            stories.place_country_code, 
            stories.place_id,
            stories.place_name, 
            stories.place_region, 
            stories.place_country, 
            stories.story_snap_id, 
            stories.story_link, 
            stories.venue_name,
            stories.detection_start_time,
            stories.detection_end_time,
            stories.time_viewed_total_day, 
            stories.heatmap_story_views_total_day,
            urls.media_url,
            PERCENT_RANK() OVER(PARTITION BY stories.place_id ORDER BY stories.heatmap_story_views_total_day DESC) AS story_views_rank,
            PERCENT_RANK() OVER(PARTITION BY stories.place_id ORDER BY urls.submit_ts DESC) AS story_freshness_rank
            FROM stories 
            LEFT JOIN urls USING(story_snap_id)
            WHERE media_url != '' 
            AND media_url is not NULL
            )
            SELECT 
            continent,
            inclusive_region,
            place_country_code,
            place_id,
            place_name,
            place_region,
            place_country,
            story_snap_id,
            story_link,
            venue_name,
            detection_start_time,
            detection_end_time,
            time_viewed_total_day,
            heatmap_story_views_total_day,
            story_views_rank,
            story_freshness_rank,
            media_url,
            story_views_rank * {view_weight} + story_freshness_rank * {freshness_weight} AS score
            FROM viral_places_stories
            
            
          """

VIDEO_REQUIRED_KEYS = [
    "short_description", "long_description", "keywords",
    "event_type", "event_scale", "event_duration",
    "event_intensity", "associated_mood", "key_objects",
    "activity_type", "contributing_context", "virality_potential"
]

PLACE_REQUIRED_KEYS = [
    "key_objects", "activity_type",
    "contributing_context",
    "short_description", "long_description",
    "keywords", "consistency", 
]

# Define columns
PLACE_MAJORITY_COLS = [
    'event_type', 'event_scale', 'event_duration', 'event_intensity', 
    'associated_mood', 
]
PLACE_LIST_COLS = [
    'keywords', 'key_objects', 'activity_type', 'contributing_context',
]

# 1. Define the complete aggregation dictionary
PLACE_AGG_DICT = {
    # Place-level attributes (keep the first value)
    'continent': 'first',
    'place_name': 'first',
    'place_country_code': 'first',
    'detection_start_time': 'first',
    'detection_end_time': 'first',
    
    # Score
    'virality_potential': 'mean',
    
    # Use majority vote for classification columns
    **{col: majority_vote for col in PLACE_MAJORITY_COLS},
    
    # LLM List Columns (Combine all unique items)
    **{col: lambda x, c=col: list(set([item for sublist in x.dropna() for item in sublist]))
       if any(x.dropna()) else [] for col in PLACE_LIST_COLS}, # Added set() for uniqueness
    
    # LLM Free-Text/Descriptive Columns (Combine text for final LLM call)
    'short_description': combine_text_list,
    'long_description': combine_text_list,
    # 'media_url': combine_text_list
}
# convenient alias
place_agg_dict = PLACE_AGG_DICT

SELECTED_COLS = ['place_id','place_name', 'place_country_code', 
       'detection_start_time', 'detection_end_time',
       'story_snap_id', 'story_link', 'venue_name', 'detection_start_time', 'detection_end_time',
       'time_viewed_total_day', 'heatmap_story_views_total_day',
       'story_views_rank', 'story_freshness_rank', 'media_url', 'score',
       'gcs_url', 'video_labels', 'video_prompt_tokens',
       'video_completion_tokens', 'short_description', 'long_description',
       'keywords', 'event_type', 'event_scale', 'event_duration',
       'event_intensity', 'associated_mood', 'key_objects',
       'activity_type', 'contributing_context', 'virality_potential'
]

STORY_COLS_RENAME = ['short_description', 'long_description',
       'keywords', 'event_type', 'event_scale', 'event_duration',
       'event_intensity', 'associated_mood', 'key_objects',
       'activity_type', 'contributing_context', 'virality_potential'
]