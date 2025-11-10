import requests
import tempfile
import os
import time
import random
import pandas as pd
from pandas.core.series import Series
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from google.cloud import storage
from google.api_core.exceptions import TooManyRequests
import re
import ast



def download_and_upload(row, 
                        *,
                        bucket: storage.Bucket,
                        bucket_name: str,
                        bucket_folder: str,
                        url_col: str,
                        id_col: str,):
    url = row[url_col]
    story_id = row[id_col]

    try:
        # Download the file
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Determine content type and file extension
        content_type = response.headers.get('content-type', 'application/octet-stream')
        ext = ""
        if 'video' in content_type:
            ext = '.mp4'
        elif 'image' in content_type:
            ext = '.jpg' if 'jpeg' in content_type else '.png'
        
        # Create a temporary file to store the download
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
        
        # Upload to GCS
        blob_name = f"{bucket_folder}/{story_id}{ext}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(temp_file.name, content_type=content_type)
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
        # Return the GCS URL
        return f"gs://{bucket_name}/{blob_name}"
    
    except Exception as e:
        print(f"Error processing {url} for story_id {story_id}: {e}")
        return None



def topk_by_score_per_place(
    df: pd.DataFrame,
    group_col: str = 'place_id',
    order_col: str = 'score',
    k: int = 3,
    filter_col = 'gcs_url',
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Return top-k rows per group, ordered by a score column (desc by default),
    keeping all original columns.

    - group_col: group key (e.g., 'place_id')
    - order_col: score column to sort within group
    - k: number of rows to keep per group
    - filter_col: if provided and present, drop rows where this col is NA or empty
    - ascending: sort direction for order_col (default False -> highest first)
    """
    if df.empty:
        return df

    df = df.copy()

    if filter_col in df.columns:
        mask = df[filter_col].notna()
        if df[filter_col].dtype == object:
            mask &= df[filter_col] != ''
        df = df[mask]

    # Validate required columns
    for c in (group_col, order_col):
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    # Sort within group and take top-k per group
    sorted_df = df.sort_values([group_col, order_col], ascending=[True, ascending])
    return (
        sorted_df
        .groupby(group_col, as_index=False, group_keys=False)
        .head(k)
    )

# Parsing helpers
def _to_float(v, default=0.0):
    """Robust float parser: handles strings, percentages, and noisy tokens."""
    try:
        s = str(v).strip()
        if s.endswith('%'):
            s = s[:-1]
        m = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", s, re.I)
        return float(m.group(0)) if m else default
    except Exception:
        return default

def _to_int(v, default=0):
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default

def _to_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v]
    if isinstance(v, str):
        return [p.strip() for p in v.split(",") if p.strip()]
    return []

def _list_to_str(v, sep: str = ', '):
    lst = _to_list(v)
    return sep.join(lst) if lst else ''

def _format_token_case(s: str) -> str:
    s = str(s).strip()
    return (s[:1].upper() + s[1:].lower()) if s else ''

def _list_to_str_cased(v, sep: str = ', '):
    lst = _to_list(v)
    if not lst:
        return ''
    return sep.join(_format_token_case(t) for t in lst if str(t).strip())

def parse_incident_json_broken(json_like) -> dict:
    result = {
        'short_description': '',
        'long_description': '',
        'keywords': '',
        'event_type': '',
        'event_scale': '',
        'event_duration': '',
        'event_intensity': '',
        'associated_mood': '',
        'key_objects_entities': '',
        'activity_type': '',
        'contributing_context': '',
        'virality_potential': 0.0,
        'detection_start_time': None,
        'detection_end_time': None,
        'place_id': '',
        'place_country_code': '',
        'consistency': 0.0,
    }

    # --- 0) If it's already a dict, normalize & return (do NOT drop it)
    if isinstance(json_like, dict):
        d = json_like
        result['short_description']     = d.get('short_description') or d.get('short description', '') or ''
        result['long_description']      = d.get('long_description')  or d.get('long description', '')  or ''
        result['event_type']            = d.get('event_type', '') or ''
        result['event_scale']           = d.get('event_scale', '') or ''
        result['event_duration']        = d.get('event_duration', '') or ''
        result['event_intensity']       = d.get('event_intensity', '') or ''
        result['associated_mood']       = d.get('associated_mood', '') or ''
        result['keywords']              = _list_to_str_cased(d.get('keywords'))
        result['key_objects_entities']  = _list_to_str_cased(d.get('key_objects_entities'))
        result['activity_type']         = _list_to_str_cased(d.get('activity_type'))
        # accept alternate key if present
        result['contributing_context']  = _list_to_str_cased(d.get('contributing_context'))
        # support multiple possible key names
        _vp = None
        for key in ('virality_potential', 'viral_potential', 'viralityPotential'):
            if d.get(key) is not None:
                _vp = d.get(key)
                break
        result['virality_potential']    = _to_float(_vp, 0.0)
        result['place_id']              = str(d.get('place_id') or '').strip()
        result['place_country_code']    = str(d.get('place_country_code') or '').strip()
        result['detection_start_time']  = pd.to_datetime(d.get('detection_start_time'), errors='coerce')
        result['detection_end_time']    = pd.to_datetime(d.get('detection_end_time'),   errors='coerce')
        if 'consistency' in d:
            result['consistency']       = _to_float(d.get('consistency'), 0.0)
        return result

    # --- 1) If it's not a string, return defaults
    if not isinstance(json_like, str):
        return result

    s = json_like.strip()

    # --- 2) Try Python literal first (handles single quotes)
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            return parse_incident_json_broken(d)  # safe now because dict branch exists
    except Exception:
        pass

    # --- 3) Regex helpers (lenient)
    # Matches quoted values, Timestamp('...'), OR bare tokens (UUIDs, codes)
    def grab_str_any(name, alt=None):
        pat = (
            r"['\"]" + re.escape(name) + r"['\"]\s*:\s*"
            r"(?:Timestamp\(['\"](?P<ts>[^'\"]+)['\"]\)"
            r"|['\"](?P<q>[^'\"]+)['\"]"
            r"|(?P<bare>[A-Za-z0-9_\-:.]+))"
        )
        m = re.search(pat, s)
        if not m and alt:
            pat_alt = (
                r"['\"]" + re.escape(alt) + r"['\"]\s*:\s*"
                r"(?:Timestamp\(['\"](?P<ts>[^'\"]+)['\"]\)"
                r"|['\"](?P<q>[^'\"]+)['\"]"
                r"|(?P<bare>[A-Za-z0-9_\-:.]+))"
            )
            m = re.search(pat_alt, s)
        if not m:
            return ''
        return (m.group('ts') or m.group('q') or m.group('bare') or '').strip()

    def grab_list(name):
        m = re.search(r"['\"]" + re.escape(name) + r"['\"]\s*:\s*\[((?:.|\n)*?)(?:\]|\Z)", s)
        if not m:
            m2 = re.search(r"['\"]" + re.escape(name) + r"['\"]\s*:\s*['\"]([^'\"]+)['\"]", s)
            return [p.strip() for p in m2.group(1).split(",")] if m2 else []
        return re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))

    def grab_float(name, default=0.0):
        pat = r"['\"]" + re.escape(name) + r"['\"]\s*:\s*['\"]?(-?\d+(?:\.\d+)?)(?:['\"])?"
        m = re.search(pat, s)
        return _to_float(m.group(1), default) if m else default

    # --- 4) Fill fields
    result['short_description']     = grab_str_any('short_description', 'short description')
    result['long_description']      = grab_str_any('long_description',  'long description')
    result['event_type']            = grab_str_any('event_type')
    result['event_scale']           = grab_str_any('event_scale')
    result['event_duration']        = grab_str_any('event_duration')
    result['event_intensity']       = grab_str_any('event_intensity')
    result['associated_mood']       = grab_str_any('associated_mood')
    # cased tokens for list-like fields
    kw = [_format_token_case(t) for t in grab_list('keywords')]
    koe = [_format_token_case(t) for t in grab_list('key_objects_entities')]
    act = [_format_token_case(t) for t in grab_list('activity_type')]
    cc = [_format_token_case(t) for t in grab_list('contributing_context')]
    result['keywords']              = ', '.join(kw)  or ''
    result['key_objects_entities']  = ', '.join(koe) or ''
    result['activity_type']         = ', '.join(act) or ''
    result['contributing_context']  = ', '.join(cc)  or ''
    # primary, with fallbacks
    vp = grab_float('virality_potential', None)
    if vp is None:
        vp = grab_float('viralityPotential', 0.0)
    result['virality_potential']    = vp

    # timestamps: get as strings (handling Timestamp('...') or quoted/bare) -> to_datetime
    ds = grab_str_any('detection_start_time')
    de = grab_str_any('detection_end_time')
    result['detection_start_time']  = pd.to_datetime(ds, errors='coerce') if ds else None
    result['detection_end_time']    = pd.to_datetime(de, errors='coerce') if de else None

    result['place_id']              = grab_str_any('place_id')
    result['place_country_code']    = grab_str_any('place_country_code')
    result['consistency']           = grab_float('consistency', 0.0)

    return result

def parse_incident_safe(x, REQUIRED_KEYS):
    try:
        d = parse_incident_json_broken(x) or {}
    except Exception:
        d = {}
    return {k: d.get(k) for k in REQUIRED_KEYS}

def majority_vote(series: Series):
    """Returns the mode (most frequent non-null value) of a pandas Series."""
    # Drop missing values and calculate the mode.
    # .mode() returns a Series of modes, use .iloc[0] to get the first one.
    return series.dropna().mode().iloc[0] if not series.dropna().empty else None

def combine_text_list(series: Series):
    """Combines all non-null text entries into a single string for LLM input."""
    return ' | '.join(series.dropna().astype(str).tolist())

def get_max_viewed_attribute(group: pd.DataFrame, rank_col: str, attribute_col: str):
    """
    Finds the row index with the maximum value in rank_col and returns 
    the corresponding attribute value from attribute_col within the group.
    """
    # Find the index of the maximum view count in the group
    max_index = group[rank_col].idxmax()
    
    # Return the corresponding attribute value from the original DataFrame
    return group.loc[max_index, attribute_col]