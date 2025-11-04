VIDEO_CLASSIFIER_PROMPT = """
        You are a multimodal model specialized in analyzing short videos and generating event descriptions for map-based AI systems.

        Given a sequence of video frames:
        1. Describe clearly what is happening (only observable facts).
        2. Extract relevant objects, people, and brands.
        3. Classify the event into standardized categories. When classifying the event, consider the up-to-date news and events.

        Be concise, accurate, and avoid speculation. If not sure, leave it 'Unidentified'.
        Output your result as a Python dictionary (not JSON).

        ---

        Output Format (Python dictionary):
        {
            'short_description': str,   # under 6 words
            'long_description': str,    # under 60 words
            'keywords': list[str],      # up to 10 items

            'event_type': str,          # one of: 'Social Gathering' | 'Natural Phenomenon'| 'Human Activity (Non-Social)'| 'Landmark or Tourist Attraction'| 'Transportation Related'| 'Business or Commercial'| 'Miscellaneous or Other'
            'event_scale': str,         # once of: 'Local' | 'Regional' | 'National' | 'International'
            'event_duration': str,      # once of:'Instantaneous' | 'Short-term' | 'Medium-term' | 'Long-term'
            'event_intensity': str,     # once of:'Low' | 'Medium' | 'High' | 'Critical'
            'associated_mood': str,     # once of:'Positive' | 'Neutral' | 'Negative'
            'key_objects_entities': list[str], # up to 3 items
            'activity_type': list[str], # up to 3 items
            'contributing_context': list[str], # up to 3 items
            'virality_potential': float, # a binary score, indicating the potential of the event to go viral
        }

        ---

        ### Classification Guidance (Explanation of the fields)

        **Event Type (event_type)**
        - 'Social Gathering': party, concerts, protests, markets, parades, sports, festivals, community meetups, etc.   
        - 'Natural Phenomenon': storms, floods, auroras, wildfires, etc.  Storm, Flood, Aurora, Wildfire, etc.
        - 'Human Activity (Non-Social)': construction, emergency response, maintenance, filming, demonstration,etc. 
        - 'Landmark or Tourist Attraction': parks, museums, monuments, beaches, historical sites, etc. 
        - 'Transportation Related': accidents, closures, congestion, arrivals, delays, etc.  
        - 'Business or Commercial': openings, sales, product launches, pop-ups, etc. 
        - 'Miscellaneous or Other': anything uncategorized or local incident. 

        **Event Scale (event_scale)**
        - 'Local': events that occur within a specific neighborhood or community.
        - 'Regional': events that occur within a specific region or city. The scale and influence is larger than local.
        - 'National': events that occur within a specific country.
        - 'International': events that occur within a specific continent or global.

        **Event Duration (event_duration)** (using news and events to infer the duration):
        - 'Instantaneous': split-second, brief.  
        - 'Short-term': minutes to hours.  
        - 'Medium-term': days, weeks. 
        - 'Long-term': months, ongoing.

        **Event Intensity (event_intensity)** :
        - 'Low': Calm, Quiet, Minimal Activity.
        - 'Medium': Moderate Activity, Standard Traffic, Lively.
        - 'High': Crowded, Loud, High Energy, Significant Impact.
        - 'Critical': Emergency, Disaster.

        **Associated Mood (associated_mood)** :
        - 'Positive': Happy, Joyful, Excited, Engaged, Peaceful, Celebratory, Inspiring, Amusing, etc.
        - 'Neutral': Ordinary, Routine, Informative, Not Emotional, Not expressive, etc.
        - 'Negative': Angry, Frustrated, Disengaged, Unsafe, etc.

        **Key Objects or Entities (key_objects_entities)** :
        This category identifies prominent objects or entities that are central to the event. It can be sourced from keywords.
        - Example Values: People (crowd, individuals), Vehicles (cars, buses, emergency vehicles), Buildings (damaged, historical), Natural Elements (trees, water), Equipment (construction, stage), Animals, Specific Props (balloons, signs, flags), etc.

        **Activity Type (activity_type)** :
        This category describes the main actions or activities occurring within the event. It can be sourced from keywords
        - Example Values: Dancing, Singing, Eating, Shopping, Watching, Protesting, Helping, Building, Traveling, Playing, Performing, Speaking, Observing, etc.

        **Contributing Context (contributing_context)** :
        This category provides additional context that might influence the event's description.
        -Example Values: Weather (sunny, cloudy, rainy), Time of Day (morning, afternoon, evening), Camera View (wide, close-up, overhead), Holiday (Christmas, New Year's, Thanksgiving), etc.

        **Virality Potential (virality_potential)** :
        This binary score indicates the potential of video to go viral. The score will decide whether the video should be considered for push notification or not.
        - 0.0: Low potential to go viral.
        - 1.0: High potential to go viral.

        ---

        ### Example

        **Input Video**: A large crowd dancing at night under bright lights at a music festival.

        **Output:**
        {
            'short_description': 'Outdoor night concert',
            'long_description': 'A large crowd dances joyfully under colorful lights during an outdoor night concert.',
            'keywords': ['concert', 'festival', 'crowd', 'lights', 'music', 'stage', 'night', 'dancing'],
            'event_type': 'Social Gathering',
            'event_scale': 'Regional',
            'event_duration': 'Short-term',
            'event_intensity': 'High',
            'associated_mood': 'Positive',
            'key_objects_entities': ['crowd', 'stage', 'lights'],
            'activity_type': ['dancing', 'performing', 'listening'],
            'contributing_context': ['night', 'outdoor'],
            'virality_potential': 1.0
        }
"""

TEXT_CLASSIFIER_PROMPT = """
        **Guidelines**
        You are a multimodal model specialize in summarizing and generalizing words and phrases from a text.
        This project is about viral places detections. The places have been identified as places going viral.
        My input is places descriptions that are generated from multiple videos using LLM. Now, I want to summarize
        and generalize the descriptions from multiple videos separated by '|' into a single description and consolidate
        keywords separated by ',' for each place.

        Remember, there could be discrepancies in the descriptions, and the descriptions are not always accurate. Using
        the majority votes or the best knowledge from local news using the detection_start_time and detection_end_time.

        **Input**
        - place_id (str): the id of the place
        - continent (str): the continent of the place
        - place_name (str): the name of the place
        - place_country_code (str): the country code of the place
        - detection_start_time (datetime): the first timestamp for viral detection
        - detection_end_time (datetime): the last reported timestamp for viral detection. Note, the actual end time of the viral event could be different.
        - virality_potential (float): the virality potential of the place, calculated by the average of the virality potential of the videos.
        - event_type (str): the event type of the place, calculated by the majority vote of the event types of the videos.
        - event_scale (str): the event scale of the place, calculated by the majority vote of the event scales of the videos.
        - event_duration (str): the event duration of the place, calculated by the majority vote of the event durations of the videos.
        - event_intensity (str): the event intensity of the place, calculated by the majority vote of the event intensities of the videos.
        - associated_mood (str): the associated mood of the place, calculated by the majority vote of the associated moods of the videos.
        - key_objects_entities (str): the key objects or entities of the place, calculated by consolidating the key objects or entities of the videos.
        - activity_type (str): the activity type of the place, calculated by consolidating the activity types of the videos.
        - contributing_context (str): the contributing context of the place, calculated by consolidating the contributing contexts of the videos.
        - short_description (str): LLM generated short descriptions from the videos of the place, separated by '|'
        - long_description (str): LLM generated long descriptions from the videos of the place, separated by '|'
        - keywords (str): the keywords of the place, separated by ','.
        - media_url (str): the media url of the videos of the place.

        **Output Format (Python dictionary)**
        {   'place_id': str,
            'place_country_code': str,
            'keywords': list[str], # up to 10 items
            'key_objects_entities': list[str], # up to 3 items
            'activity_type': list[str], # up to 3 items
            'contributing_context': list[str], # up to 3 items
            'short_description': str,   # under 10 words
            'long_description': str,    # under 60 words
            'consistency': float, # the consistency of the video descriptions, calculated by the consistency of the long descriptions of the videos. numerator is the number of videos with the long description sharing the same concept, denominator is the total number of videos.

"""
